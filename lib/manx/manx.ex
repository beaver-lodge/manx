defmodule Manx do
  @moduledoc """
  `Manx` is a MLIR backend for the `Nx`. It mainly targets TOSA/Linalg dialect and will generate LLVM/CUDA/Vulkan code for different configurations.
  """

  @enforce_keys [:memory]
  defstruct memory: nil, device: :host

  @behaviour Nx.Backend

  alias Nx.Tensor, as: T
  alias __MODULE__, as: B

  @impl Nx.Backend
  def init(keyword) do
    ctx = Beaver.MLIR.Context.create()
    Beaver.Diagnostic.attach(ctx)
    keyword ++ [ctx: ctx]
  end

  @impl Nx.Backend
  def constant(out, constant, backend_options) do
    binary_tensor = Nx.BinaryBackend.constant(out, constant, [])
    Nx.BinaryBackend.backend_transfer(binary_tensor, __MODULE__, backend_options)
  end

  @impl Nx.Backend
  def from_binary(%T{shape: shape, type: type} = tensor, binary, backend_options) do
    shape = Tuple.to_list(shape)
    device = Keyword.get(backend_options, :device, :host)

    memory =
      Beaver.Native.Memory.new(
        binary,
        sizes: shape,
        type: type
      )

    memory |> Manx.MemrefAllocator.add()
    put_in(tensor.data, %B{memory: memory, device: device})
  end

  @impl Nx.Backend
  def to_binary(%T{shape: _shape, data: %B{memory: memory}} = tensor, limit) do
    Beaver.Native.Memory.aligned(memory)
    |> Beaver.Native.OpaquePtr.to_binary(limit * div(element_size(tensor), 8))
  end

  @impl Nx.Backend
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    tensor
    |> to_binary(min(limit, Nx.size(tensor)))
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
  end

  defp element_size(%T{type: {_, size}}), do: size

  @impl Nx.Backend
  def backend_copy(tensor, Nx.Tensor, backend_options) do
    backend_copy(tensor, Nx.BinaryBackend, backend_options)
  end

  # TODO: Support direct transfers without going through Elixir
  def backend_copy(
        %T{shape: shape, data: %B{memory: memory}} = tensor,
        backend,
        backend_options
      ) do
    binary_len = Enum.reduce(Tuple.to_list(shape), 1, &*/2) * div(element_size(tensor), 8)

    backend.from_binary(
      tensor,
      Beaver.Native.Memory.aligned(memory)
      |> Beaver.Native.OpaquePtr.to_binary(binary_len),
      backend_options
    )
  end

  @impl Nx.Backend
  def backend_transfer(
        %T{data: %B{memory: memory}} = tensor,
        backend,
        backend_options
      ) do
    if backend == __MODULE__ do
      # TODO: support tensor on device memory like CUDA
      tensor
    else
      tensor = backend_copy(tensor, backend, backend_options)

      with :ok <- Manx.MemrefAllocator.delete(memory) do
        tensor
      else
        :already_deallocated -> raise "called on deleted or donated buffer"
      end
    end
  end

  @impl Nx.Backend
  def backend_deallocate(%T{data: %B{memory: memory}}) do
    memory |> Manx.MemrefAllocator.delete()
  end

  @doc """
  Create a new tensor of null ptr memref. This should be used as as the return tensor of JIT function.
  """
  def tensor_of_null_memref(%T{shape: shape, type: _type} = tensor) do
    shape = Tuple.to_list(shape)

    memory = Beaver.Native.Memory.new(nil, sizes: shape, type: Beaver.Native.F32)

    put_in(tensor.data, %B{memory: memory})
  end

  def tensor_of_null_memref(tuple) when is_tuple(tuple) do
    for t <- tuple |> Tuple.to_list() do
      tensor_of_null_memref(t)
    end
    |> List.to_tuple()
  end

  # TODO: check if argument is returned by JIT function
  @doc """
  Add returned memref to the allocator.
  """
  def add_allocated_memory(%T{data: %B{memory: memory}} = tensor) do
    memory = memory |> Beaver.Native.Memory.own_allocated()
    memory |> Manx.MemrefAllocator.add()
    put_in(tensor.data.memory, memory)
  end

  def add_allocated_memory(tuple) when is_tuple(tuple) do
    for t <- Tuple.to_list(tuple) do
      add_allocated_memory(t)
    end
    |> List.to_tuple()
  end

  require Nx.Defn.Expr
  ## JIT callbacks

  @impl Nx.Backend
  def concatenate(out, tensors, axis) do
    out = Nx.to_template(out)

    expr_fun = fn tensors ->
      Nx.Defn.Expr.concatenate(out, Tuple.to_list(tensors), axis)
    end

    jit(expr_fun, [List.to_tuple(tensors)])
  end

  @impl Nx.Backend
  def slice(out, tensor, start_indices, lengths, strides) do
    out = Nx.to_template(out)

    if Enum.all?(start_indices, &is_integer/1) do
      expr_fun = fn tensor ->
        Nx.Defn.Expr.slice(out, tensor, start_indices, lengths, strides)
      end

      jit(expr_fun, [tensor])
    else
      expr_fun = fn tensor, start_indices ->
        Nx.Defn.Expr.slice(out, tensor, Tuple.to_list(start_indices), lengths, strides)
      end

      jit(expr_fun, [tensor, List.to_tuple(start_indices)])
    end
  end

  @impl Nx.Backend
  def put_slice(out, tensor, start_indices, slice) do
    out = Nx.to_template(out)

    if Enum.all?(start_indices, &is_integer/1) do
      expr_fun = fn tensor, slice ->
        Nx.Defn.Expr.put_slice(out, tensor, start_indices, slice)
      end

      jit(expr_fun, [tensor, slice])
    else
      expr_fun = fn tensor, start_indices, slice ->
        Nx.Defn.Expr.put_slice(out, tensor, Tuple.to_list(start_indices), slice)
      end

      jit(expr_fun, [tensor, List.to_tuple(start_indices), slice])
    end
  end

  @impl Nx.Backend
  def optional(_name, args, fun) do
    # Here we take the leading tensor arguments and pass them as JIT arguments
    {tensors, rest} = Enum.split_while(args, &is_struct(&1, Nx.Tensor))

    wrapper_fun = fn tensors ->
      tensors = Tuple.to_list(tensors)
      apply(fun, tensors ++ rest)
    end

    jit(wrapper_fun, [List.to_tuple(tensors)])
  end

  binary_ops =
    [:add, :subtract, :multiply, :pow, :remainder, :divide, :atan2, :min, :max, :quotient] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
      [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] ++
      [:logical_and, :logical_or, :logical_xor]

  unary_ops =
    [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tan] ++
      [:cosh, :sinh, :tanh, :acos, :asin, :atan, :acosh, :asinh, :atanh] ++
      [:sqrt, :rsqrt, :cbrt, :is_nan, :is_infinity, :erf, :erfc, :erf_inv] ++
      [:abs, :bitwise_not, :ceil, :conjugate, :floor, :negate, :round, :sign] ++
      [:count_leading_zeros, :population_count, :real, :imag]

  callbacks =
    [
      {:eye, [:backend_options], []},
      {:iota, [:axis, :backend_options], []},
      {:as_type, [:tensor], [:tensor]},
      {:bitcast, [:tensor], [:tensor]},
      {:reshape, [:tensor], [:tensor]},
      {:squeeze, [:tensor, :axes], [:tensor]},
      {:broadcast, [:tensor, :shape, :axes], [:tensor]},
      {:transpose, [:tensor, :axes], [:tensor]},
      {:pad, [:tensor, :pad_value, :padding_config], [:tensor, :pad_value]},
      {:reverse, [:tensor, :axes], [:tensor]},
      {:dot, [:left, :c1, :b1, :right, :c2, :b2], [:left, :right]},
      {:clip, [:tensor, :min, :max], [:tensor, :min, :max]},
      {:take, [:tensor, :indices, :axis], [:tensor, :indices]},
      {:take_along_axis, [:tensor, :indices, :axis], [:tensor, :indices]},
      {:gather, [:input, :indices, :opts], [:input, :indices]},
      {:select, [:pred, :on_true, :on_false], [:pred, :on_true, :on_false]},
      {:conv, [:tensor, :kernel, :opts], [:tensor, :kernel]},
      {:all, [:tensor, :opts], [:tensor]},
      {:any, [:tensor, :opts], [:tensor]},
      {:sum, [:tensor, :opts], [:tensor]},
      {:product, [:tensor, :opts], [:tensor]},
      {:reduce_max, [:tensor, :opts], [:tensor]},
      {:reduce_min, [:tensor, :opts], [:tensor]},
      {:argmax, [:tensor, :opts], [:tensor]},
      {:argmin, [:tensor, :opts], [:tensor]},
      {:reduce, [:tensor, :acc, :opts, :fun], [:tensor, :acc]},
      {:window_reduce, [:tensor, :acc, :shape, :opts, :fun], [:tensor, :acc]},
      {:window_sum, [:tensor, :shape, :opts], [:tensor]},
      {:window_product, [:tensor, :shape, :opts], [:tensor]},
      {:window_max, [:tensor, :shape, :opts], [:tensor]},
      {:window_min, [:tensor, :shape, :opts], [:tensor]},
      {:map, [:tensor, :opts, :fun], [:tensor]},
      {:sort, [:tensor, :opts], [:tensor]},
      {:argsort, [:tensor, :opts], [:tensor]},
      {:window_scatter_max, [:tensor, :source, :init_value, :window_dims, :opts],
       [:tensor, :source, :init_value]},
      {:window_scatter_min, [:tensor, :source, :init_value, :window_dims, :opts],
       [:tensor, :source, :init_value]},
      {:indexed_add, [:tensor, :indices, :updates, :opts], [:tensor, :indices, :updates]},
      {:indexed_put, [:tensor, :indices, :updates, :opts], [:tensor, :indices, :updates]},
      {:lu, [:tensor, :opts], [:tensor]},
      {:triangular_solve, [:a, :b, :opts], [:a, :b]},
      {:fft, [:tensor, :opts], [:tensor]},
      {:ifft, [:tensor, :opts], [:tensor]}
    ] ++
      for(op <- binary_ops, do: {op, [:left, :right], [:left, :right]}) ++
      for(op <- unary_ops, do: {op, [:tensor], [:tensor]})

  for {name, args, tensor_args} <- callbacks do
    args = Enum.map(args, &Macro.var(&1, __MODULE__))
    tensor_args = Enum.map(tensor_args, &Macro.var(&1, __MODULE__))

    @impl Nx.Backend
    def unquote(name)(out, unquote_splicing(args)) do
      out = Nx.to_template(out)

      expr_fun = fn unquote_splicing(tensor_args) ->
        Nx.Defn.Expr.unquote(name)(out, unquote_splicing(args))
      end

      jit(expr_fun, [unquote_splicing(tensor_args)])
    end
  end

  def jit(function, args, options \\ []) do
    Nx.Defn.jit_apply(function, args, Keyword.put(options, :compiler, __MODULE__))
  end
end
