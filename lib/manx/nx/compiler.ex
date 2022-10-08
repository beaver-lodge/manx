defmodule Manx.Compiler do
  alias Beaver.MLIR
  import MLIR.Sigils
  import Beaver, only: :macros
  require Beaver.MLIR
  alias Beaver.MLIR.Dialect.{Func}
  require Func
  @behaviour Nx.Defn.Compiler
  @impl true
  def __jit__(key, vars, fun, [args], _options) do
    # call fun to generated tree
    tree = fun.(vars)

    info = Function.info(key)
    uniq = info |> Keyword.get(:uniq)
    module = info |> Keyword.get(:module)
    name = info |> Keyword.get(:name)

    symbol =
      Module.concat([module, name, "#{uniq}"])
      |> Atom.to_string()

    # generate ir
    entry_types =
      Enum.reduce(vars, [], fn
        tuple, acc when is_tuple(tuple) ->
          acc ++ Enum.map(Tuple.to_list(tuple), &Manx.Defn.gen_type/1)

        t, acc ->
          acc ++ [Manx.Defn.gen_type(t)]
      end)

    arg0 = args |> List.first()

    module_attrs =
      if arg0 do
        with %Manx{device: :vulkan} <- arg0.data do
          [
            "spv.target_env":
              ~a"#spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spv.resource_limits<>>"
          ]
        else
          _ -> []
        end
      else
        []
      end

    ctx = MLIR.Context.create()

    ir =
      mlir ctx: ctx do
        module(module_attrs) do
          function_type =
            Type.function(
              entry_types,
              Manx.Defn.gen_root_types(tree)
            )

          Func.func manx_main(
                      sym_name: "\"#{symbol}\"",
                      function_type: function_type
                    ) do
            region do
              locs = List.duplicate(MLIR.Location.unknown(), length(entry_types))

              entry =
                MLIR.Block.create(
                  entry_types |> Enum.map(&Beaver.Deferred.create(&1, MLIR.__CONTEXT__())),
                  locs |> Enum.map(&Beaver.Deferred.create(&1, MLIR.__CONTEXT__()))
                )

              root = Manx.Defn.gen_op(%Manx.Defn.Env{block: entry, ctx: ctx}, tree)

              mlir block: entry do
                case root do
                  ret = %Beaver.MLIR.Value{} ->
                    Func.return(ret) >>> []

                  tuple_ret when is_tuple(tuple_ret) ->
                    Func.return(Tuple.to_list(tuple_ret)) >>> []
                end
              end

              MLIR.__REGION__()
              |> Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(entry)
            end
          end
        end
      end

    {llvm_ir, libs} =
      if arg0 do
        case arg0.data do
          %Nx.BinaryBackend{} ->
            {Manx.Lowering.CPU.lower(ir), []}

          %Manx{device: device} ->
            case device do
              :host ->
                {Manx.Lowering.CPU.lower(ir), []}

              :vulkan ->
                {:ok, llvm_lib_dir} = Beaver.LLVM.Config.lib_dir()

                {Manx.Lowering.Vulkan.lower(ir),
                 [
                   llvm_lib_dir |> Path.join("libvulkan-runtime-wrappers.dylib"),
                   llvm_lib_dir |> Path.join("libmlir_runner_utils.dylib")
                 ]}
            end
        end
      else
        # If all args are materialized as constants, let's assume it's all cpu
        {Manx.Lowering.CPU.lower(ir), []}
      end

    jit =
      llvm_ir
      |> MLIR.ExecutionEngine.create!(shared_lib_paths: libs)

    # invoke jit and setting return for tree
    tree_return =
      tree
      |> Manx.tensor_of_null_memref()
      |> invoke(args, jit, symbol)

    MLIR.CAPI.mlirContextDestroy(ctx)
    [tree_return]
  end

  @doc """
  Invoke MLIR JIT with Nx tensors. If there are tuples their memrefs will be packed into a single C struct.
  """

  def invoke(return, args, jit, symbol) do
    # pack the tensor tuples into a C struct
    jit_args =
      [return_struct | _] =
      [return | args]
      |> Enum.map(&memref_from_tensor/1)

    if List.improper?(jit_args), do: raise("jit arguments is not a proper list")

    MLIR.ExecutionEngine.invoke!(
      jit,
      symbol,
      jit_args |> Enum.map(&Beaver.Native.Memory.descriptor_ptr/1)
    )

    # unpack the C struct into tensor tuples
    populate_tensor_from_memref(return, return_struct)
    |> Manx.add_allocated_memory()
  end

  @doc """
  - If it is a tensor, return a memref
  - If it is a tuple, recursively pack them into one struct.
  """
  def memref_from_tensor(%Nx.Tensor{data: %Manx{memory: memory}}), do: memory

  def memref_from_tensor(
        %Nx.Tensor{
          data: %Nx.BinaryBackend{state: binary}
        } = tensor
      ) do
    Manx.from_binary(tensor, binary, []) |> memref_from_tensor
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
    kind = first.descriptor.kind

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
            kind: element_kind
          }
        }

        {acc ++ [populate_tensor_from_memref(x, mem)], offset + size}
      end)

    tensors |> List.to_tuple()
  end
end
