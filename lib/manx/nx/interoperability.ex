defmodule Manx.Nx.Interoperability do
  @moduledoc """
  Functions for interoperability between Elixir/NX and LLVM/MLIR. For instance the data transfer between NX tensor and MemRef.
  """

  alias Beaver.MLIR

  @doc """
  - If it is a tensor, return a memref
  - If it is a tuple, recursively pack them into one struct.
  """
  def memref_from_tensor(f) when is_function(f), do: f.() |> memref_from_tensor
  def memref_from_tensor(%Nx.Tensor{data: %Manx{memory: memory}}), do: memory

  def memref_from_tensor(
        %Nx.Tensor{
          data: %Nx.BinaryBackend{state: binary}
        } = tensor
      ) do
    Manx.from_binary(tensor, binary, []) |> memref_from_tensor
  end

  def memref_from_tensor(%Nx.Tensor{shape: shape, data: %Nx.TemplateBackend{}}) do
    # TODO: generate a magical deadbeef pointer for this
    Beaver.Native.Memory.new(nil, sizes: shape |> Tuple.to_list(), type: Beaver.Native.F32)
  end

  def memref_from_tensor({}) do
    raise "can't extract memref from an empty tuple"
  end

  def memref_from_tensor(tuple) when is_tuple(tuple) do
    mems =
      Tuple.to_list(tuple)
      |> Enum.map(&memref_from_tensor/1)

    # TODO: support array of memref descriptor of different kinds
    first = mems |> List.first()
    kind = first.descriptor.descriptor_kind

    refs =
      mems
      |> Enum.map(fn %Beaver.Native.Memory{descriptor: %Beaver.Native.Memory.Descriptor{ref: ref}} ->
        ref
      end)

    # TODO: add a raw NIF beaver_raw_create_heterogeneous_array, using union maybe
    mut_array = Beaver.Native.forward(kind, :mut_array, [refs])

    struct!(Beaver.Native.Array,
      element_kind: kind,
      ref: mut_array
    )
  end

  @doc """
  - If it is a tensor, return a memref
  - If it is a tuple, recursively unpack each member from the nested struct.
  """
  def populate_tensor_from_memref(%Nx.Tensor{data: %Manx{}} = tensor, memory) do
    %{tensor | data: %Manx{memory: memory}}
  end

  def populate_tensor_from_memref(
        tuple,
        %Beaver.Native.Array{element_kind: element_kind} = nested_struct
      )
      when is_tuple(tuple) do
    nested_struct_ptr = nested_struct |> Beaver.Native.Memory.descriptor_ptr()

    {tensors, _offset} =
      Enum.reduce(tuple |> Tuple.to_list(), {[], 0}, fn x, {acc, offset} ->
        {ref, size} =
          Beaver.Native.OpaquePtr.to_resource(
            element_kind,
            nested_struct_ptr,
            offset
          )

        mem = %Beaver.Native.Memory{
          descriptor: %Beaver.Native.Memory.Descriptor{
            ref: ref,
            descriptor_kind: element_kind
          }
        }

        {acc ++ [populate_tensor_from_memref(x, mem)], offset + size}
      end)

    tensors |> List.to_tuple()
  end

  def loc_from_stack_trace({:current_stacktrace, frames}, ctx) do
    loc_from_stack_trace(frames, ctx)
  end

  def loc_from_stack_trace(frames, ctx) do
    stacktrace_locs =
      for {_, _, _, f} <- frames do
        f
      end
      |> Enum.map(&[name: to_string(&1[:file]), line: &1[:line], ctx: ctx])
      |> Enum.reject(&String.starts_with?(&1[:name], "lib/process.ex"))
      |> Enum.map(&MLIR.Location.file(&1))

    MLIR.CAPI.mlirLocationFusedGet(
      ctx,
      length(stacktrace_locs),
      Beaver.Native.array(stacktrace_locs, MLIR.Location),
      MLIR.Attribute.null()
    )
  end
end
