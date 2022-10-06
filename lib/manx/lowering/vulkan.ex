defmodule Manx.Lowering.Vulkan do
  alias Beaver.MLIR
  import MLIR.{Transforms, Conversion}

  def lower(op) do
    op
    |> MLIR.Operation.verify!(dump_if_fail: true)
    |> canonicalize
    |> MLIR.Pass.Composer.nested("func.func", fn pm ->
      MLIR.Pass.pipeline!(pm, "tosa-make-broadcastable")
      MLIR.Pass.pipeline!(pm, "llvm-request-c-wrappers")
      MLIR.Pass.pipeline!(pm, "tosa-layerwise-constant-fold")
    end)
    |> cse
    |> tosa_to_arith
    |> tosa_to_tensor()
    |> convert_tensor_to_linalg()
    |> MLIR.Pass.Composer.nested("func.func", [
      tosa_to_linalg(),
      linalg_fuse_elementwise_ops(),
      linalg_bufferize(),
      convert_linalg_to_parallel_loops(),
      gpu_map_parallel_loops()
    ])
    |> MLIR.Pass.Composer.pipeline("arith-bufferize,func-bufferize")
    |> convert_parallel_loops_to_gpu()
    |> gpu_launch_sink_index_computations()
    |> gpu_kernel_outlining()
    |> MLIR.Pass.Composer.nested("gpu.module", fn pm ->
      MLIR.CAPI.mlirOpPassManagerAddOwnedPass(pm, lower_affine())
      npm = MLIR.CAPI.mlirOpPassManagerGetNestedUnder(pm, MLIR.StringRef.create("gpu.func"))
      MLIR.CAPI.mlirOpPassManagerAddOwnedPass(npm, __MODULE__.PutSPVAttrPass.create())
    end)
    |> MLIR.Pass.Composer.nested("func.func", fn pm ->
      MLIR.Pass.pipeline!(pm, "tensor-bufferize")
    end)
    |> MLIR.Pass.Composer.nested("func.func", [
      linalg_bufferize(),
      convert_linalg_to_loops(),
      lower_affine(),
      convert_math_to_llvm(),
      convert_arith_to_llvm(),
      convert_scf_to_cf(),
      "arith-expand",
      "memref-expand"
    ])
    |> MLIR.Pass.Composer.nested("gpu.module", fn pm ->
      # MLIR.CAPI.mlirOpPassManagerAddOwnedPass(pm, map_memref_spirv_storage_class())
      npm = MLIR.CAPI.mlirOpPassManagerGetNestedUnder(pm, MLIR.StringRef.create("gpu.func"))

      for p <- [
            convert_math_to_spirv(),
            # convert_arith_to_spirv(),
            convert_cf_to_spirv()
            # convert_tensor_to_spirv(),
            # convert_vector_to_spirv(),
            # convert_func_to_spirv(),
            # convert_memref_to_spirv(),
            # convert_scf_to_spirv()
          ] do
        MLIR.CAPI.mlirOpPassManagerAddOwnedPass(npm, p)
      end
    end)
    |> convert_gpu_to_spirv()
    |> MLIR.Pass.Composer.nested("spv.module", fn pm ->
      MLIR.Pass.pipeline!(pm, "spirv-lower-abi-attrs")
      MLIR.Pass.pipeline!(pm, "spirv-update-vce")
    end)
    |> convert_gpu_launch_to_vulkan_launch
    |> convert_memref_to_llvm
    |> MLIR.Pass.Composer.nested("func.func", fn pm ->
      MLIR.Pass.pipeline!(pm, "llvm-request-c-wrappers")
    end)
    |> convert_complex_to_standard()
    |> convert_vector_to_llvm
    |> convert_complex_to_llvm()
    |> convert_func_to_llvm
    |> reconcile_unrealized_casts
    |> launch_func_to_vulkan
    |> MLIR.Pass.Composer.run!(dump_if_fail: false, print: Manx.Flags.print_ir?())
  end
end
