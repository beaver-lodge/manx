defmodule Manx.Compiler do
  use Beaver
  alias Beaver.MLIR
  import MLIR.Sigils
  import Beaver, only: :macros
  require Beaver.MLIR
  alias Beaver.MLIR.Dialect.{Func}
  require Func
  @behaviour Nx.Defn.Compiler

  defp eval_arg(f) when is_function(f), do: f.()
  defp eval_arg(list) when is_list(list), do: Enum.map(list, &eval_arg/1)
  defp eval_arg(a), do: a

  defp runtime_libs() do
    case LLVMConfig.lib_dir() do
      {:ok, llvm_lib_dir} ->
        [
          llvm_lib_dir |> Path.join("libmlir_c_runner_utils.dylib")
        ]

      _ ->
        []
    end
  end

  defp vulkan_runtime_libs() do
    case LLVMConfig.lib_dir() do
      {:ok, llvm_lib_dir} ->
        [
          llvm_lib_dir |> Path.join("libvulkan-runtime-wrappers.dylib")
        ]

      _ ->
        []
    end
  end

  # Invoke MLIR JIT with Nx tensors. If there are tuples their memrefs will be packed into a single C struct.
  defp invoke(return, args, jit, symbol) do
    import Manx.Nx.Interoperability
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

  defp module_attrs([tensor | _]), do: module_attrs(tensor)

  defp module_attrs(%Nx.Tensor{data: %Manx{device: :vulkan}}) do
    [
      "spirv.target_env":
        ~a"#spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>"
    ]
  end

  defp module_attrs(_), do: []

  defp lower(ir, []), do: {Manx.Lowering.CPU.lower(ir), runtime_libs()}
  defp lower(ir, [tensor | _]), do: lower(ir, tensor)

  defp lower(ir, %Nx.Tensor{data: %Nx.BinaryBackend{}}) do
    {Manx.Lowering.CPU.lower(ir), runtime_libs()}
  end

  defp lower(ir, %Nx.Tensor{data: %Manx{device: :host}}) do
    {Manx.Lowering.CPU.lower(ir), runtime_libs()}
  end

  defp lower(ir, %Nx.Tensor{data: %Manx{device: :vulkan}}) do
    {Manx.Lowering.Vulkan.lower(ir), vulkan_runtime_libs()}
  end

  @doc false
  @impl Nx.Defn.Compiler
  def __jit__(key, vars, fun, args_list, options) do
    __compile__(key, vars, fun, options).(args_list)
  end

  @doc false
  @impl Nx.Defn.Compiler
  def __compile__(key, vars, fun, _options) do
    # call fun to generate expression tree
    tree = fun.(vars)
    info = Function.info(key)
    uniq = info |> Keyword.get(:uniq)
    module = info |> Keyword.get(:module)
    name = info |> Keyword.get(:name)
    symbol = Module.concat([module, name, "#{uniq}"]) |> Atom.to_string()

    # generate ir
    entry_types =
      Enum.reduce(vars, [], fn
        tuple, acc when is_tuple(tuple) ->
          acc ++ Enum.map(Tuple.to_list(tuple), &Manx.Defn.gen_type/1)

        t, acc ->
          acc ++ [Manx.Defn.gen_type(t)]
      end)

    fn args_list ->
      args_list = args_list |> Enum.map(&eval_arg/1)

      for args <- args_list do
        ctx = MLIR.Context.create()
        Beaver.Diagnostic.attach(ctx)

        ir =
          mlir ctx: ctx do
            module(module_attrs(args)) do
              function_type =
                Type.function(
                  entry_types,
                  Manx.Defn.gen_root_types(tree)
                )

              stacktrace_loc =
                Process.info(self(), :current_stacktrace)
                |> Manx.Nx.Interoperability.loc_from_stack_trace(ctx)

              Func.func manx_main(
                          sym_name: "\"#{symbol}\"",
                          function_type: function_type,
                          loc: stacktrace_loc
                        ) do
                region do
                  locs = List.duplicate(stacktrace_loc, length(entry_types))

                  entry =
                    MLIR.Block.create(
                      entry_types |> Enum.map(&Beaver.Deferred.create(&1, Beaver.Env.context())),
                      locs |> Enum.map(&Beaver.Deferred.create(&1, Beaver.Env.context()))
                    )

                  mlir block: entry do
                    case Manx.Defn.gen_op(%Manx.Defn.Env{block: entry, ctx: ctx}, tree) do
                      ret = %Beaver.MLIR.Value{} ->
                        Func.return(ret, loc: stacktrace_loc) >>> []

                      tuple_ret when is_tuple(tuple_ret) ->
                        Func.return(Tuple.to_list(tuple_ret), loc: stacktrace_loc) >>> []
                    end
                  end

                  Beaver.Env.region()
                  |> Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(entry)
                end
              end
            end
          end

        case lower(ir, args) do
          {{:ok, mod}, libs} ->
            jit =
              mod
              |> MLIR.ExecutionEngine.create!(shared_lib_paths: libs)

            # invoke jit and setting return for tree
            tree_return =
              tree
              |> Manx.tensor_of_null_memref()
              |> invoke(args, jit, symbol)

            MLIR.Module.destroy(mod)
            MLIR.Context.destroy(ctx)
            tree_return

          {{:error, msg}, _} ->
            MLIR.Context.destroy(ctx)
            raise msg
        end
      end
    end
  end

  @doc false
  @impl Nx.Defn.Compiler
  def __stream__(
        _key,
        _input,
        _acc,
        _vars,
        _fun,
        _args_list,
        _opts
      ),
      do: raise("not implemented")

  @doc false
  @impl Nx.Defn.Compiler
  def __to_backend__(_keyword), do: raise("not implemented")

  @doc false
  @impl Nx.Defn.Compiler
  def __partitions_options__(_keyword), do: raise("not implemented")
end
