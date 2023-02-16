defmodule Manx.Lowering.CPU do
  alias Beaver.MLIR
  import MLIR.{Transforms, Conversion}

  defp one_shot(op) do
    op
    |> MLIR.Operation.verify!()
    |> canonicalize
    |> MLIR.Pass.Composer.nested("func.func", "tosa-make-broadcastable")
    |> MLIR.Pass.Composer.nested("func.func", "tosa-layerwise-constant-fold")
    |> cse
    |> tosa_to_scf
    |> tosa_to_arith
    |> tosa_to_tensor()
    |> convert_tensor_to_linalg()
    |> MLIR.Pass.Composer.nested("func.func", [
      tosa_to_linalg_named(),
      tosa_to_linalg(),
      "empty-tensor-to-alloc-tensor",
      linalg_fuse_elementwise_ops(),
      "tosa-layerwise-constant-fold",
      lower_affine()
    ])
    |> MLIR.Pass.Composer.nested("func.func", "empty-tensor-to-alloc-tensor")
    |> MLIR.Pass.Composer.append("one-shot-bufferize{allow-return-allocs create-deallocs=false}")
    |> MLIR.Pass.Composer.append("func-bufferize,arith-bufferize")
    |> MLIR.Pass.Composer.nested("func.func", [
      convert_linalg_to_loops(),
      convert_scf_to_cf(),
      "arith-expand",
      convert_arith_to_llvm(),
      convert_math_to_llvm()
    ])
    |> MLIR.Pass.Composer.nested("func.func", "llvm-request-c-wrappers")
    |> convert_math_to_libm
    |> convert_complex_to_standard()
    |> convert_vector_to_llvm
    |> MLIR.Pass.Composer.nested("func.func", "expand-strided-metadata,memref-expand")
    |> convert_memref_to_llvm
    |> convert_complex_to_llvm()
    |> convert_func_to_llvm
    |> reconcile_unrealized_casts
    |> MLIR.Pass.Composer.run(print: Manx.Flags.print_ir?(), debug: false)
  end

  defp do_lower(op) do
    op
    |> MLIR.Operation.verify!()
    |> MLIR.Pass.Composer.nested("func.func", "tosa-make-broadcastable")
    |> MLIR.Pass.Composer.nested("func.func", "tosa-layerwise-constant-fold")
    |> cse
    |> tosa_to_scf
    |> tosa_to_arith
    |> tosa_to_tensor()
    |> convert_tensor_to_linalg()
    |> MLIR.Pass.Composer.nested("func.func", [
      tosa_to_linalg_named(),
      tosa_to_linalg(),
      linalg_fuse_elementwise_ops(),
      "tosa-layerwise-constant-fold",
      linalg_bufferize(),
      convert_linalg_to_loops(),
      "affine-expand-index-ops",
      lower_affine(),
      convert_math_to_llvm(),
      convert_arith_to_llvm(),
      convert_scf_to_cf(),
      "arith-expand"
    ])
    |> MLIR.Pass.Composer.nested("func.func", "empty-tensor-to-alloc-tensor")
    |> MLIR.Pass.Composer.append("arith-bufferize,func-bufferize")
    |> MLIR.Pass.Composer.nested("func.func", "tensor-bufferize")
    |> MLIR.Pass.Composer.nested("func.func", "llvm-request-c-wrappers")
    |> MLIR.Pass.Composer.nested("func.func", "expand-strided-metadata,memref-expand")
    |> convert_math_to_libm
    |> convert_complex_to_standard()
    |> convert_vector_to_llvm
    |> convert_memref_to_llvm
    |> convert_complex_to_llvm()
    |> convert_func_to_llvm
    |> reconcile_unrealized_casts
    |> MLIR.Pass.Composer.run(print: Manx.Flags.print_ir?())
  end

  @doc """
  Run passes to compile IR generated from Nx expressions, mostly in TOSA and some LinAlg. The results should be in LLVM.
  """
  def lower(op, opts \\ []) do
    one_shot = opts[:one_shot] || false
    # canonicalize it first to fold operations of index but result type fixed in Nx expression
    case op
         |> canonicalize
         |> MLIR.Pass.Composer.run(print: Manx.Flags.print_ir?()) do
      {:ok, op} ->
        if one_shot do
          one_shot(op)
        else
          do_lower(op)
        end

      result ->
        result
    end
  end
end
