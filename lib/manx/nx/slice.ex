defmodule Manx.Slice do
  @moduledoc false
  alias Manx.Defn.Env
  use Beaver
  alias Beaver.MLIR
  import MLIR.Sigils
  import Beaver, only: :macros
  require Beaver.MLIR
  alias MLIR.{Type, Attribute}
  alias MLIR.Dialect.{TOSA, Arith, Tensor, SCF, MemRef}

  def static_slice(
        %Env{block: block, ctx: ctx, gen_op: gen_op, gen_type: gen_type} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            op: :slice,
            args: [tensor, start_indices, lengths, strides]
          }
        } = t
      ) do
    mlir block: block, ctx: ctx do
      input_value = gen_op.(env, tensor)

      sizes =
        for {start, length, stride} <- Enum.zip([start_indices, lengths, strides]) do
          limit = start + length
          Integer.floor_div(limit - 1 - start + stride, stride)
        end
        |> Attribute.dense_array(Beaver.Native.I64)

      offsets = Attribute.dense_array(start_indices, Beaver.Native.I64)

      if Enum.all?(strides, &Kernel.==(&1, 1)) do
        TOSA.slice(input_value, start: offsets, size: sizes) >>> gen_type.(t)
      else
        Tensor.extract_slice(input_value,
          static_offsets: offsets,
          static_sizes: sizes,
          static_strides: Attribute.dense_array(strides, Beaver.Native.I64),
          operand_segment_sizes: ODS.operand_segment_sizes([1, 0, 0, 0])
        ) >>> gen_type.(t)
      end
    end
  end

  def dynamic_slice(
        %Env{block: block, ctx: ctx, gen_op: gen_op, gen_type: gen_type} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            op: :slice,
            args: [tensor, start_indices, lengths, strides]
          }
        } = t
      ) do
    mlir block: block, ctx: ctx do
      input_value = gen_op.(env, tensor)

      start_indices =
        for {{start, length}, index} <- Enum.zip([start_indices, lengths]) |> Enum.with_index() do
          start_value = gen_op.(env, start)
          extracted = Tensor.extract(start_value) >>> gen_type.(start.type)

          start_index =
            case start.type do
              {:s, _} ->
                Arith.index_castui(extracted) >>> Type.index()

              {:f, _} ->
                Arith.index_cast(extracted) >>> Type.index()
            end

          mn = Arith.constant(value: Attribute.index(0)) >>> Type.index()
          dim = Arith.constant(value: Attribute.index(index)) >>> Type.index()
          mx = Tensor.dim(input_value, dim) >>> Type.index()
          size = Arith.constant(value: Attribute.index(length)) >>> Type.index()
          mx = Arith.subi(mx, size) >>> Type.index()
          start_index = Arith.maxsi(start_index, mn) >>> Type.index()
          Arith.minsi(start_index, mx) >>> Type.index()
        end

      sizes = lengths |> Attribute.dense_array(Beaver.Native.I64)
      strides = strides |> Attribute.dense_array(Beaver.Native.I64)

      offsets =
        Attribute.dense_array(
          List.duplicate(
            Beaver.MLIR.CAPI.mlirShapedTypeGetDynamicStrideOrOffset(),
            length(lengths)
          ),
          Beaver.Native.I64
        )

      Tensor.extract_slice(
        input_value,
        start_indices,
        static_offsets: offsets,
        static_sizes: sizes,
        static_strides: strides,
        operand_segment_sizes: ODS.operand_segment_sizes([1, length(lengths), 0, 0])
      ) >>>
        gen_type.(t)
    end
  end
end
