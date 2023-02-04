Nx.Defn.default_options(compiler: Manx.Compiler, default_backends: {Manx, device: :vulkan})
defmodule ManxVulkanAttention do
  import Nx.Defn


  defn softmax(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t), axes: [-1], keep_axes: true)
  end

  defn batched_dot(t1, t2) do
    Nx.dot(t1, [2], [0], t2, [1], [0])
  end

  @doc """
  dim is the dimension of each head
  """
  defn scaled_dot_product_attention(query, key, value, dim) do
    score = Nx.dot(query, [2], [0], key, [2], [0]) / Nx.sqrt(dim)
    attn = softmax(score)
    Nx.dot(attn, [2], [0], value, [1], [0])
  end
end


query = Nx.iota({4, 3, 2}, type: {:f, 32}) |> Nx.divide(10.0)
query = Nx.backend_transfer(query, {Manx, device: :vulkan})
key = Nx.iota({4, 3, 2}, type: {:f, 32}) |> Nx.divide(10.0)
key = Nx.backend_transfer(key, {Manx, device: :vulkan})
value = Nx.iota({4, 3, 2}, type: {:f, 32}) |> Nx.divide(10.0)
value = Nx.backend_transfer(value, {Manx, device: :vulkan})

for i <- 1..100 do
  r = try do
    ManxVulkanAttention.scaled_dot_product_attention(query, key, value, 12)
    :ok
  rescue e ->
    :error
  end
  case r do
    :ok ->
      nil
      raise "ok"
    :error ->
      nil
  end
end
