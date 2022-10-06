defmodule Manx.AttentionTest do
  use ExUnit.Case, async: true
  import Nx.Defn
  import Manx.Assert

  @moduletag :nx
  @moduletag :attention
  setup do
    Nx.Defn.default_options(compiler: Manx.Compiler)
    :ok
  end

  # original implementation from: https://github.com/sooftware/attentions/blob/master/attentions.py
  describe "attention" do
    defn(softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t), axes: [-1], keep_axes: true))

    defn(batched_dot(t1, t2), do: Nx.dot(t1, [2], [0], t2, [1], [0]))

    @doc """
    dim is the dimension of each head
    """
    defn scaled_dot_product_attention(dim, query, key, value) do
      score = Nx.dot(query, [2], [0], key, [2], [0]) / Nx.sqrt(dim)
      attn = softmax(score)
      Nx.dot(attn, [2], [0], value, [1], [0])
    end

    test "dot product attention" do
      # do a divide to prevent overflow
      query = Nx.iota({4, 3, 2}, type: {:f, 32}) |> Nx.divide(10.0)
      key = Nx.iota({4, 3, 2}, type: {:f, 32}) |> Nx.divide(10.0)
      value = Nx.iota({4, 3, 2}, type: {:f, 32}) |> Nx.divide(10.0)

      assert_all_close(
        scaled_dot_product_attention(12, query, key, value),
        Nx.tensor([
          [[0.2008, 0.3008], [0.2038, 0.3038], [0.2069, 0.3069]],
          [[0.8100, 0.9100], [0.8131, 0.9131], [0.8161, 0.9161]],
          [[1.4192, 1.5192], [1.4222, 1.5222], [1.4253, 1.5253]],
          [[2.0283, 2.1283], [2.0313, 2.1313], [2.0343, 2.1343]]
        ])
      )
    end
  end
end
