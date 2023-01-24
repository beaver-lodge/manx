defmodule Manx.Lowering.CPU do
  alias Beaver.MLIR
  import MLIR.{Transforms, Conversion}

  @doc """
  Run passes to compile IR generated from Nx expressions, mostly in TOSA and some LinAlg. The results should be in LLVM.
  """
  def lower(op) do
    op
    |> MLIR.Operation.verify!(dump_if_fail: true)
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
      linalg_fuse_elementwise_ops(),
      "tosa-layerwise-constant-fold",
      linalg_bufferize(),
      convert_linalg_to_loops(),
      lower_affine(),
      convert_math_to_llvm(),
      convert_arith_to_llvm(),
      convert_scf_to_cf(),
      "arith-expand",
      "memref-expand"
    ])
    |> MLIR.Pass.Composer.nested("func.func", "tensor-bufferize")
    |> MLIR.Pass.Composer.append("arith-bufferize,func-bufferize")
    |> MLIR.Pass.Composer.nested("func.func", "llvm-request-c-wrappers")
    |> convert_math_to_libm
    |> convert_complex_to_standard()
    |> convert_vector_to_llvm
    |> convert_memref_to_llvm
    |> convert_complex_to_llvm()
    |> convert_func_to_llvm
    |> reconcile_unrealized_casts
    |> MLIR.Pass.Composer.run!(dump_if_fail: true, print: Manx.Flags.print_ir?())
  end
end
