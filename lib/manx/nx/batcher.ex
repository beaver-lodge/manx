defmodule Manx.Nx.Batcher do
  alias Beaver.MLIR
  require Beaver.MLIR
  alias MLIR.Attribute
  defstruct [:tensor, :contract_axes, :batch_axes]

  def from_args([
        %Nx.Tensor{} = a,
        contract_axes1,
        batch_axes1,
        %Nx.Tensor{} = b,
        contract_axes2,
        batch_axes2
      ]) do
    batched_a = %Manx.Nx.Batcher{
      tensor: a,
      contract_axes: contract_axes1,
      batch_axes: batch_axes1
    }

    batched_b = %Manx.Nx.Batcher{
      tensor: b,
      contract_axes: contract_axes2,
      batch_axes: batch_axes2
    }

    {batched_a, batched_b}
  end

  # [[CONTRACT DIMS]...[BATCH DIMS]...[OUTER DIMS]...]
  defmodule AffineMapAcc do
    defstruct exprs: [], contract_index: 0, batch_index: nil, outer_index: nil
  end

  defp gen_input_affine_map(
         %__MODULE__{
           tensor: %{shape: shape},
           batch_axes: batch_axes,
           contract_axes: contract_axes
         },
         {maps, outer_index},
         output_rank: output_rank
       ) do
    rank = tuple_size(shape)

    acc =
      Enum.reduce(
        Range.new(0, rank - 1, 1),
        %AffineMapAcc{outer_index: outer_index, batch_index: length(contract_axes)},
        fn dim, acc ->
          case {dim in contract_axes, dim in batch_axes} do
            {true, false} ->
              %{
                acc
                | exprs: acc.exprs ++ [MLIR.AffineMap.dim(acc.contract_index)],
                  contract_index: acc.contract_index + 1
              }

            {false, true} ->
              %{
                acc
                | exprs: acc.exprs ++ [MLIR.AffineMap.dim(acc.batch_index)],
                  batch_index: acc.batch_index + 1
              }

            {false, false} ->
              %{
                acc
                | exprs: acc.exprs ++ [MLIR.AffineMap.dim(acc.outer_index)],
                  outer_index: acc.outer_index + 1
              }
          end
        end
      )

    {maps ++ [MLIR.AffineMap.create(output_rank, 0, acc.exprs)], acc.outer_index}
  end

  defp gen_output_affine_map(contract_axes_length, rank) when is_integer(rank) do
    out_rank = contract_axes_length + rank

    exprs =
      for dim <- Range.new(0, out_rank - 1, 1), dim >= contract_axes_length do
        MLIR.AffineMap.dim(dim)
      end

    MLIR.AffineMap.create(out_rank, 0, exprs)
  end

  def gen_indexing_maps(
        %__MODULE__{batch_axes: batch_axes_a, contract_axes: contract_axes_a} = a,
        %__MODULE__{batch_axes: batch_axes_b, contract_axes: contract_axes_b} = b,
        c
      )
      when length(contract_axes_a) == length(contract_axes_b) and
             length(batch_axes_a) == length(batch_axes_b) do
    contract_axes_length = length(contract_axes_a)
    batch_axes_length = length(batch_axes_b)

    output_rank = tuple_size(c.shape) + contract_axes_length

    {input_maps, _} =
      Enum.reduce(
        [a, b],
        {[], contract_axes_length + batch_axes_length},
        &gen_input_affine_map(&1, &2, output_rank: output_rank)
      )

    Enum.concat(input_maps, [
      gen_output_affine_map(contract_axes_length, tuple_size(c.shape))
    ])
    |> Enum.map(&MLIR.Attribute.affine_map/1)
    |> Attribute.array()
  end

  def gen_iterator_types(
        %__MODULE__{contract_axes: contract_axes_a},
        %__MODULE__{contract_axes: contract_axes_b},
        c
      )
      when length(contract_axes_a) == length(contract_axes_b) do
    contract_iterator_types =
      Attribute.string("reduction")
      |> List.duplicate(length(contract_axes_a))

    outer_iterator_types =
      Attribute.string("parallel")
      |> List.duplicate(tuple_size(c.shape))

    Enum.concat(contract_iterator_types, outer_iterator_types)
    |> Attribute.array()
  end
end
