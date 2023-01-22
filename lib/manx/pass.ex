defmodule Manx.Lowering.Vulkan.PutSPVAttrPass do
  alias Beaver.MLIR
  import MLIR.Sigils

  use Beaver.MLIR.Pass, on: "gpu.func"

  @impl true
  def run(op) do
    [
      "gpu.kernel": Beaver.MLIR.Attribute.unit(),
      "spirv.entry_point_abi": ~a{#spirv.entry_point_abi<workgroup_size = [16, 1, 1]>}
    ]
    |> Enum.each(fn {name, attr} ->
      ctx = MLIR.CAPI.mlirOperationGetContext(op)
      attr = Beaver.Deferred.create(attr, ctx)
      MLIR.CAPI.mlirOperationSetAttributeByName(op, MLIR.StringRef.create(name), attr)
    end)

    :ok
  end
end
