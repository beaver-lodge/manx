defmodule Manx.Lowering.Vulkan.PutSPVAttrPass do
  alias Beaver.MLIR
  import MLIR.Sigils
  alias Beaver.MLIR.Dialect.GPU
  use Beaver.MLIR.Pass, on: GPU.Func

  def run(op) do
    [
      "gpu.kernel": Beaver.MLIR.Attribute.unit(),
      "spirv.entry_point_abi":
        ~a{#spirv.entry_point_abi<local_size = dense<[1, 1, 1]> : vector<3xi32>>}
    ]
    |> Enum.each(fn {name, attr} ->
      ctx = MLIR.CAPI.mlirOperationGetContext(op)
      attr = Beaver.Deferred.create(attr, ctx)
      MLIR.CAPI.mlirOperationSetAttributeByName(op, MLIR.StringRef.create(name), attr)
    end)

    :ok
  end
end
