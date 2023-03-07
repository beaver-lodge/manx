defmodule Manx.Linalg do
  import Beaver.MLIR.Sigils
  alias Beaver.MLIR
  alias Beaver.MLIR.Attribute
  @moduledoc false

  def expand_for_output(input_shape, output_shape)
      when tuple_size(output_shape) >= tuple_size(input_shape) do
    output_rank = tuple_size(output_shape)
    rank = tuple_size(input_shape)
    expanded = List.duplicate(1, output_rank - rank) ++ Tuple.to_list(input_shape)
    List.to_tuple(expanded)
  end

  defp gen_identity(shape), do: &MLIR.CAPI.mlirAffineMapMultiDimIdentityGet(&1, tuple_size(shape))

  defp gen_broadcast_minor_identity(in_shape, out_shape) do
    rank = tuple_size(out_shape)
    rank_diff = rank - tuple_size(in_shape)

    zipped =
      in_shape
      |> expand_for_output(out_shape)
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(out_shape))

    exprs =
      for {{in_dim, out_dim}, index} <- zipped |> Enum.with_index(), index >= rank_diff do
        case {in_dim, out_dim} do
          {1, out_dim} when out_dim != 1 ->
            0

          _ ->
            MLIR.AffineMap.dim(index)
        end
      end

    MLIR.AffineMap.create(rank, 0, exprs)
  end

  defp maps_to_attr(maps) do
    maps
    |> Enum.map(&MLIR.Attribute.affine_map/1)
    |> Attribute.array()
  end

  # unary, always identity
  defp do_gen_indexing_maps(shape, shape) do
    do_gen_indexing_maps([shape], shape)
  end

  defp do_gen_indexing_maps([shape], shape) do
    gen_identity(shape)
    |> List.duplicate(2)
  end

  # binary+, might broadcast
  defp do_gen_indexing_maps(input_shapes, out_shape)
       when is_list(input_shapes) and length(input_shapes) > 1 do
    Enum.map(input_shapes, &gen_broadcast_minor_identity(&1, out_shape)) ++
      [gen_identity(out_shape)]
  end

  def gen_indexing_maps(input_shapes, out_shape) do
    do_gen_indexing_maps(input_shapes, out_shape) |> maps_to_attr
  end

  def gen_indexing_maps(out_shape) do
    [gen_identity(out_shape)] |> maps_to_attr
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
