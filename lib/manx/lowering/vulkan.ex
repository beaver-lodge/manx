defmodule Manx.Lowering.Vulkan do
  alias Beaver.MLIR
  import MLIR.{Transforms, Conversion}

  def lower(op) do
    op
    |> MLIR.Operation.verify!(dump_if_fail: true)
    |> canonicalize
    |> MLIR.Pass.Composer.nested(
      "func.func",
      ~w{tosa-make-broadcastable llvm-request-c-wrappers tosa-layerwise-constant-fold}
    )
    |> cse
    |> tosa_to_arith
    |> tosa_to_tensor()
    |> convert_tensor_to_linalg()
    |> MLIR.Pass.Composer.nested("func.func", [
      tosa_to_linalg_named(),
      tosa_to_linalg(),
      linalg_generalize_named_ops(),
      linalg_fuse_elementwise_ops(),
      linalg_bufferize(),
      convert_linalg_to_parallel_loops(),
      gpu_map_parallel_loops()
    ])
    |> MLIR.Pass.Composer.append("arith-bufferize,func-bufferize")
    |> convert_parallel_loops_to_gpu()
    |> gpu_launch_sink_index_computations()
    |> gpu_kernel_outlining()
    |> MLIR.Pass.Composer.nested("gpu.module", [
      {
        :nested,
        "gpu.func",
        [
          lower_affine(),
          MLIR.ExternalPass.create(__MODULE__.PutSPVAttrPass)
        ]
      }
    ])
    |> MLIR.Pass.Composer.nested("func.func", "tensor-bufferize")
    |> MLIR.Pass.Composer.nested("gpu.module", [
      {
        :nested,
        "gpu.func",
        [
          convert_memref_to_spirv(),
          convert_math_to_spirv(),
          convert_arith_to_spirv(),
          convert_cf_to_spirv(),
          convert_tensor_to_spirv(),
          convert_vector_to_spirv(),
          convert_func_to_spirv(),
          convert_scf_to_spirv()
        ]
      }
    ])
    |> convert_gpu_to_spirv()
    |> MLIR.Pass.Composer.nested(
      "spirv.module",
      ~w{spirv-lower-abi-attrs spirv-update-vce}
    )
    |> convert_gpu_launch_to_vulkan_launch
    |> MLIR.Pass.Composer.append("expand-strided-metadata")
    |> MLIR.Pass.Composer.append("finalize-memref-to-llvm")
    |> MLIR.Pass.Composer.nested("func.func", "llvm-request-c-wrappers")
    |> convert_complex_to_standard()
    |> convert_vector_to_llvm
    |> convert_complex_to_llvm()
    |> convert_func_to_llvm
    |> reconcile_unrealized_casts
    |> launch_func_to_vulkan
    |> MLIR.Pass.Composer.run(dump_if_fail: false, print: Manx.Flags.print_ir?())
  end
end
