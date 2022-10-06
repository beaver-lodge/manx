defmodule Manx.VulkanExprTest do
  use ExUnit.Case, async: true
  import Nx.Defn
  import Manx.Assert

  @moduletag :nx
  @moduletag :vulkan
  setup do
    Nx.Defn.default_options(compiler: Manx.Compiler, default_backends: {Manx, device: :vulkan})
    :ok
  end

  describe "unary float ops" do
    @float_tensor Nx.tensor([1.0, 2.0, 3.0])
    defn unary_sin(t) do
      Nx.sin(t)
    end

    test "sin" do
      t_vulkan = Nx.backend_transfer(@float_tensor, {Manx, device: :vulkan})
      assert_all_close(unary_sin(t_vulkan), evaluate(&unary_sin/1, [@float_tensor]))
    end

    @int_tensor_a Nx.tensor([1, 2, 3], type: {:u, 32})
    @int_tensor_b Nx.tensor([4, 5, 6], type: {:u, 32})
    defn binary_add(a, b) do
      Nx.add(a, b)
    end

    test "add" do
      a = Nx.backend_transfer(@int_tensor_a, {Manx, device: :vulkan})
      b = Nx.backend_transfer(@int_tensor_b, {Manx, device: :vulkan})
      assert [5, 7, 9] == binary_add(a, b) |> Nx.to_flat_list()
    end

    defn unary_add(a) do
      Nx.add(a, @int_tensor_b)
    end

    test "add a constant" do
      a = Nx.backend_transfer(@int_tensor_a, {Manx, device: :vulkan})
      assert [5, 7, 9] == unary_add(a) |> Nx.to_flat_list()
    end
  end
end
