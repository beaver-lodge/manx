defmodule Manx.Type do
  require Beaver.MLIR
  alias Beaver.MLIR.Type

  @moduledoc """
  Helper functions for defining functions and operations.
  """

  def gen_type({:u, size}), do: Type.i(size)
  def gen_type({:s, size}), do: Type.i(size)
  def gen_type({:f, size}), do: Type.f(size)
  def gen_type({:c, size}), do: Type.complex(Type.f(div(size, 2)))

  def gen_type(%Nx.Tensor{shape: shape, type: type}) do
    Tuple.to_list(shape)
    |> Type.ranked_tensor(gen_type(type))
  end

  def gen_type(tuple) when is_tuple(tuple) do
    Tuple.to_list(tuple)
    |> Enum.map(&gen_type/1)
    |> Type.tuple()
  end
end
