defmodule Manx.Linalg do
  import Beaver.MLIR.Sigils
  alias Beaver.MLIR
  alias Beaver.MLIR.Attribute
  @moduledoc false
  defp gen_affine_map(shape) do
    import MLIR.AffineMap
    rank = tuple_size(shape)

    exprs =
      shape
      |> Tuple.to_list()
      |> Enum.with_index()
      |> Enum.map(fn
        {1, _index} -> 0
        {dim_size, index} when dim_size > 1 -> dim(index)
      end)

    MLIR.AffineMap.create(rank, 0, exprs)
  end

  def expand_for_output(input_shape, output_shape)
      when tuple_size(output_shape) >= tuple_size(input_shape) do
    output_rank = tuple_size(output_shape)
    rank = tuple_size(input_shape)
    expanded = List.duplicate(1, output_rank - rank) ++ Tuple.to_list(input_shape)
    List.to_tuple(expanded)
  end

  def gen_indexing_maps(out_shape) do
    [
      gen_affine_map(out_shape)
    ]
    |> Enum.map(&MLIR.Attribute.affine_map/1)
    |> Attribute.array()
  end

  def gen_indexing_maps(input1_shape, out_shape) do
    [
      expand_for_output(input1_shape, out_shape) |> gen_affine_map(),
      gen_affine_map(out_shape)
    ]
    |> Enum.map(&MLIR.Attribute.affine_map/1)
    |> Attribute.array()
  end

  def gen_indexing_maps(
        input1_shape,
        input2_shape,
        out_shape
      ) do
    [
      expand_for_output(input1_shape, out_shape) |> gen_affine_map(),
      expand_for_output(input2_shape, out_shape) |> gen_affine_map(),
      gen_affine_map(out_shape)
    ]
    |> Enum.map(&MLIR.Attribute.affine_map/1)
    |> Attribute.array()
  end

  def gen_iterator_types({}, {}) do
    ~a{[]}
  end

  def gen_iterator_types({_}, {_}) do
    ~a{[#linalg.iterator_type<parallel>]}
  end

  def gen_iterator_types(input, output) when input == output do
    case tuple_size(input) do
      1 ->
        ~a{[#linalg.iterator_type<parallel>]}

      2 ->
        ~a{[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>]}
    end
  end

  def gen_iterator_types({}, {}, _output) do
    ~a{[]}
  end

  def gen_iterator_types(input1, _input2, output) do
    input1 = expand_for_output(input1, output)

    case tuple_size(input1) do
      1 ->
        ~a{[#linalg.iterator_type<parallel>]}

      2 ->
        ~a{[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>]}
    end
  end

  def gen_iterator_types(output) do
    for _ <- output |> Tuple.to_list() do
      ~a{#linalg.iterator_type<parallel>}
    end
    |> Attribute.array()
  end
end
