defmodule Manx.Defn do
  alias __MODULE__.Env
  use Beaver
  alias Beaver.MLIR
  import MLIR.Sigils
  import Beaver, only: :macros
  require Beaver.MLIR
  alias MLIR.{Type, Attribute}
  alias MLIR.Dialect.{TOSA, Linalg, Arith, Tensor, Bufferization, Math, SCF, MemRef}

  defdelegate gen_type(tensor), to: Manx.Type

  @doc """
  In upstream MLIR, there is no lower-able Op packing multiple values into a tuple.
  If the Nx root type is a tuple, it should be converted to multi-results.
  This function should always return a list of types
  """
  def gen_root_types(tuple) when is_tuple(tuple) do
    Tuple.to_list(tuple)
    |> Enum.map(&gen_type/1)
  end

  def gen_root_types(type), do: [gen_type(type)]

  defp gen_expand(
         _env,
         value,
         %{shape: in_shape},
         %{shape: out_shape}
       )
       when tuple_size(in_shape) == tuple_size(out_shape) do
    value
  end

  defp gen_expand(
         %Env{block: block, ctx: ctx},
         value,
         %{type: type, shape: in_shape} = input_t,
         %{shape: out_shape} = _output_t
       ) do
    mlir block: block, ctx: ctx do
      shape = Manx.Linalg.expand_for_output(in_shape, out_shape)
      t = %{input_t | type: type, shape: shape}
      rank_diff = tuple_size(out_shape) - tuple_size(in_shape)

      pairs =
        Range.new(0, tuple_size(in_shape) - 1, 1)
        |> Enum.map(fn i -> [i, i + rank_diff] end)

      Tensor.expand_shape(value, reassociation: Tensor.reassociation(pairs)) >>> gen_type(t)
    end
  end

  def gen_op(
        %Env{block: block},
        %Nx.Tensor{
          data: %Nx.Defn.Expr{op: :parameter, args: [pos]}
        }
      )
      when is_integer(pos) do
    arg_cnt = Beaver.Walker.arguments(block) |> Enum.count()

    if pos >= arg_cnt do
      raise "argument ##{pos} out of bound, argument count: #{arg_cnt}"
    end

    arg = block |> Beaver.MLIR.CAPI.mlirBlockGetArgument(pos)

    if MLIR.is_null(arg) do
      raise "argument ##{pos} not found"
    end

    arg
  end

  def gen_op(
        %Env{block: block, ctx: ctx},
        %Nx.Tensor{
          data: %Nx.Defn.Expr{op: :constant, args: [:nan]},
          shape: {},
          type: {:f, 32}
        } = t
      ) do
    mlir block: block, ctx: ctx do
      TOSA.const({:value, ~a{dense<0x7F800001> : tensor<f32>}}) >>> gen_type(t)
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx},
        %Nx.Tensor{
          data: %Nx.Defn.Expr{op: :constant, args: [:infinity]},
          shape: {},
          type: {:f, 32}
        } = t
      ) do
    mlir block: block, ctx: ctx do
      TOSA.const({:value, ~a{dense<0x7F800000> : tensor<f32>}}) >>>
        gen_type(t)
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx},
        %Nx.Tensor{
          data: %Nx.Defn.Expr{op: :constant, args: [:neg_infinity]},
          shape: {},
          type: {:f, 32}
        } = t
      ) do
    mlir block: block, ctx: ctx do
      _r =
        TOSA.const({:value, ~a{dense<0xFF800000> : tensor<f32>}}) >>>
          gen_type(t)
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx},
        %Nx.Tensor{
          data: %Nx.Defn.Expr{op: :constant, args: [value]},
          shape: {}
        } = t
      )
      when is_integer(value) or is_float(value) do
    mlir block: block, ctx: ctx do
      t_str = gen_type(t) |> Beaver.Deferred.create(ctx) |> MLIR.to_string()

      TOSA.const({:value, ~a{dense<#{value}> : #{t_str}}}) >>>
        gen_type(t)
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx},
        %Nx.Tensor{
          data: %Nx.Defn.Expr{op: :constant, args: [%Complex{im: im, re: re}]},
          type: {:c, 64}
        } = t
      ) do
    mlir block: block, ctx: ctx do
      t_str = gen_type(t) |> Beaver.Deferred.create(ctx) |> MLIR.to_string()

      Arith.constant({:value, ~a[dense<(#{re}, #{im})> : #{t_str}]}) >>>
        gen_type(t)
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx},
        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            args: [%Nx.Tensor{data: %Nx.BinaryBackend{state: binary}}],
            op: :tensor
          }
        } = t
      ) do
    mlir block: block, ctx: ctx do
      tensor_attr =
        MLIR.CAPI.mlirDenseElementsAttrRawBufferGet(
          gen_type(t) |> Beaver.Deferred.create(ctx),
          byte_size(binary),
          Beaver.Native.c_string(binary) |> Beaver.Native.Array.as_opaque()
        )

      if MLIR.Attribute.is_null(tensor_attr), do: raise("fail to parse tensor dense elements")

      TOSA.const({:value, tensor_attr}) >>> gen_type(t)
    end
  end

  # unary tosa
  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{data: %Nx.Defn.Expr{op: op, args: [input1]}} = t
      )
      when op in [
             :negate,
             :abs,
             :bitwise_not,
             :exp,
             :logical_not,
             :log,
             :tanh,
             :rsqrt,
             :is_nan,
             :is_infinity,
             :sigmoid
           ] do
    mlir block: block, ctx: ctx do
      input1_value = gen_op(env, input1)
      input1_value = TOSA.cast(input1_value) >>> gen_type(%{input1 | type: t.type})

      case op do
        :negate ->
          TOSA.negate(input1_value) >>> gen_type(t)

        :abs ->
          TOSA.abs(input1_value) >>> gen_type(t)

        :bitwise_not ->
          TOSA.bitwise_not(input1_value) >>> gen_type(t)

        :logical_not ->
          input1_value = TOSA.cast(input1_value) >>> gen_type(%{t | type: {:u, 1}})
          result = TOSA.logical_not(input1_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(result) >>> gen_type(t)

        :exp ->
          TOSA.exp(input1_value) >>> gen_type(t)

        :log ->
          TOSA.log(input1_value) >>> gen_type(t)

        :tanh ->
          TOSA.tanh(input1_value) >>> gen_type(t)

        :rsqrt ->
          TOSA.rsqrt(input1_value) >>> gen_type(t)

        :sigmoid ->
          TOSA.sigmoid(input1_value) >>> gen_type(t)

        :is_nan ->
          c = TOSA.equal(input1_value, input1_value) >>> gen_type(%{t | type: {:u, 1}})
          c = TOSA.logical_not(c) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :is_infinity ->
          input1_value = gen_op(env, input1)
          input1_type_str = gen_type(input1) |> Beaver.Deferred.create(ctx) |> MLIR.to_string()

          inf =
            TOSA.const({:value, ~a{dense<0x7F800000> : #{input1_type_str}}}) >>> gen_type(input1)

          abs = TOSA.abs(input1_value) >>> gen_type(input1)
          equal = TOSA.equal(inf, abs) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(equal) >>> gen_type(t)
      end
    end
  end

  def gen_op(
        env,
        %Nx.Tensor{shape: {}, data: %Nx.Defn.Expr{op: :all, args: [%{shape: {}} = input1, _]}}
      ) do
    gen_op(env, input1)
  end

  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            op: :squeeze,
            args: [input, _axes]
          }
        } = t
      ) do
    mlir block: block, ctx: ctx do
      input_value = gen_op(env, input)
      source_type = gen_type(input) |> Beaver.Deferred.create(ctx)
      target_type = gen_type(t) |> Beaver.Deferred.create(ctx)
      reassociation = Tensor.reassociation_for_reshape(source_type, target_type)

      if MLIR.is_null(reassociation) do
        raise "fail to create reassociation"
      end

      Tensor.collapse_shape(input_value, reassociation: reassociation) >>> target_type
    end
  end

  def gen_op(
        %Env{} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            op: :slice,
            args: [_tensor, start_indices, _lengths, _strides]
          }
        } = t
      ) do
    env = %Env{env | gen_op: &gen_op/2, gen_type: &gen_type/1}

    if Enum.all?(start_indices, &is_integer/1) do
      Manx.Slice.static_slice(env, t)
    else
      Manx.Slice.dynamic_slice(env, t)
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            op: op,
            args: [%{shape: in_shape} = input1, [axes: axes, keep_axes: keep_axes]]
          }
        } = t
      )
      when is_list(axes) and op in [:all, :sum] do
    mlir block: block, ctx: ctx do
      input1 = gen_op(env, input1)

      input1 =
        case op do
          :all ->
            TOSA.cast(input1) >>> gen_type(%{t | shape: in_shape, type: {:u, 1}})

          :sum ->
            input1
        end

      {in_shape, mlir_value} =
        Enum.reduce(
          axes,
          {Tuple.to_list(in_shape), input1},
          fn axis, {in_shape, mlir_value} ->
            out_shape = List.replace_at(in_shape, axis, 1)

            reduce_attr = [axis: Attribute.integer(Type.i64(), axis)]

            reduced =
              case op do
                :all ->
                  TOSA.reduce_all(mlir_value, reduce_attr) >>>
                    gen_type(%{t | shape: List.to_tuple(out_shape), type: {:u, 1}})

                :sum ->
                  TOSA.reduce_sum(mlir_value, reduce_attr) >>>
                    gen_type(%{t | shape: List.to_tuple(out_shape)})
              end

            {out_shape, reduced}
          end
        )

      mlir_value = TOSA.cast(mlir_value) >>> gen_type(%{t | shape: List.to_tuple(in_shape)})

      if keep_axes do
        mlir_value
      else
        Tensor.collapse_shape(mlir_value, reassociation: Tensor.reassociation([])) >>> gen_type(t)
      end
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{
          data:
            %Nx.Defn.Expr{
              op: op,
              args: [%{shape: in_shape} = input1, [axes: nil, keep_axes: keep_axes]]
            } = expr
        } = t
      )
      when op in [:sum, :all] do
    # if axes is nil, replace it with a list of every axis
    mlir block: block, ctx: ctx do
      rank = tuple_size(in_shape)
      axes = Range.new(0, rank - 1, 1) |> Enum.to_list()

      expr = %{
        expr
        | args: [input1, [axes: axes, keep_axes: keep_axes]]
      }

      gen_op(env, %{t | data: expr})
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            op: :conjugate,
            args: [%Nx.Tensor{type: {:c, 64}} = complex_tensor]
          },
          shape: {}
        } = t
      ) do
    alias MLIR.Dialect.Complex

    mlir block: block, ctx: ctx do
      complex_tensor = gen_op(env, complex_tensor)
      complex_element = Tensor.extract(complex_tensor) >>> Type.complex(Type.f32())
      conjugate_element = Complex.conj(complex_element) >>> Type.complex(Type.f32())

      conjugate_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ODS.operand_segment_sizes([0, 0, 0])) >>>
          gen_type(t)

      Tensor.insert(conjugate_element, conjugate_tensor) >>>
        gen_type(t)
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{op: :conjugate, args: [%Nx.Tensor{} = real_tensor]},
          shape: {},
          type: complex_type = {:c, 64}
        } = t
      ) do
    alias MLIR.Dialect.Complex

    mlir block: block, ctx: ctx do
      real_tensor = gen_op(env, real_tensor)
      real_tensor = TOSA.cast(real_tensor) >>> Type.ranked_tensor([], Type.f32())
      real = Tensor.extract(real_tensor) >>> Type.f32()

      conjugate_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ODS.operand_segment_sizes([0, 0, 0])) >>>
          gen_type(t)

      imaginary = Arith.constant(value: Attribute.float(Type.f32(), 0.0)) >>> Type.f32()

      complex_element_t = gen_type(complex_type)
      complex_element = Complex.create(real, imaginary) >>> complex_element_t
      conjugate_element = Complex.conj(complex_element) >>> complex_element_t

      _ = Tensor.insert(conjugate_element, conjugate_tensor) >>> gen_type(t)
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{op: :conjugate, args: [complex_tensor]},
          shape: shape
        } = t
      ) do
    alias MLIR.Dialect.Complex

    mlir block: block, ctx: ctx do
      element_cnt = Enum.reduce(Tuple.to_list(shape), 1, &*/2)
      complex_tensor = gen_op(env, complex_tensor)
      lower = Arith.constant(value: Attribute.integer(Type.index(), 0)) >>> Type.index()
      upper = Arith.constant(value: Attribute.integer(Type.index(), element_cnt)) >>> Type.index()
      step = Arith.constant(value: Attribute.integer(Type.index(), 1)) >>> Type.index()

      conjugate_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ODS.operand_segment_sizes([0, 0, 0])) >>>
          gen_type(t)

      conjugate_memref =
        Bufferization.to_memref(conjugate_tensor) >>>
          Type.memref([2], Type.complex(Type.f32()))

      SCF.for [lower, upper, step] do
        region do
          block inner(index >>> Type.index()) do
            complex_element = Tensor.extract(complex_tensor, index) >>> Type.complex(Type.f32())
            conjugate_element = Complex.conj(complex_element) >>> Type.complex(Type.f32())
            MemRef.store(conjugate_element, conjugate_memref, index) >>> []
            SCF.yield() >>> []
          end
        end
      end >>> []

      conjugate_tensor
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            op: :imag,
            args: [%Nx.Tensor{type: {:c, 64}, shape: in_shape} = in_tensor]
          },
          shape: out_shape
        } = t
      ) do
    alias MLIR.Dialect.Complex

    mlir block: block, ctx: ctx do
      in_tensor = gen_op(env, in_tensor)

      out_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ODS.operand_segment_sizes([0, 0, 0])) >>>
          gen_type(t)

      Linalg.generic [
        in_tensor,
        out_tensor,
        operand_segment_sizes: ODS.operand_segment_sizes([1, 1]),
        indexing_maps: Manx.Linalg.gen_indexing_maps(in_shape, out_shape),
        iterator_types: Manx.Linalg.gen_iterator_types(in_shape, out_shape)
      ] do
        region do
          block bb0(arg0 >>> Type.complex(Type.f32()), arg1 >>> Type.f(32)) do
            %MLIR.Value{} = arg1
            im = Complex.im(arg0) >>> Type.f32()
            Linalg.yield([im]) >>> []
          end
        end
      end >>> gen_type(t)
    end
  end

  # unary linalg
  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{type: type, data: %Nx.Defn.Expr{op: op, args: [input]}} = t
      )
      when op in [
             :population_count,
             :count_leading_zeros,
             :cos,
             :sin,
             :sqrt,
             :tan,
             :erf,
             :cbrt,
             :expm1,
             :log1p,
             :bitcast,
             :atan
           ] do
    mlir block: block, ctx: ctx do
      input_value = gen_op(env, input)
      input_value = TOSA.cast(input_value) >>> gen_type(t)

      out_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ODS.operand_segment_sizes([0, 0, 0])) >>>
          gen_type(t)

      Linalg.generic [
        input_value,
        out_tensor,
        operand_segment_sizes: ODS.operand_segment_sizes([1, 1]),
        indexing_maps: Manx.Linalg.gen_indexing_maps(input.shape, t.shape),
        iterator_types: Manx.Linalg.gen_iterator_types(input.shape, t.shape)
      ] do
        region do
          block bb0(arg0 >>> gen_type(type), out >>> gen_type(type)) do
            %MLIR.Value{} = out

            result =
              case op do
                :population_count ->
                  Math.ctpop(arg0) >>> gen_type(type)

                :count_leading_zeros ->
                  Math.ctlz(arg0) >>> gen_type(type)

                :cos ->
                  Math.cos(arg0) >>> gen_type(type)

                :sin ->
                  Math.sin(arg0) >>> gen_type(type)

                :sqrt ->
                  Math.sqrt(arg0) >>> gen_type(type)

                :tan ->
                  Math.tan(arg0) >>> gen_type(type)

                :erf ->
                  Math.erf(arg0) >>> gen_type(type)

                :bitcast ->
                  Arith.bitcast(arg0) >>> gen_type(type)

                :cbrt ->
                  abs =
                    case type do
                      {i_type, _} when i_type in [:i, :s] ->
                        Math.absi(arg0) >>> gen_type(type)

                      {f_type, _} when f_type in [:f] ->
                        Math.absf(arg0) >>> gen_type(type)
                    end

                  third =
                    Arith.constant(value: Attribute.float(gen_type(type), 0.333333343)) >>>
                      gen_type(type)

                  pow = Math.powf(abs, third) >>> gen_type(type)
                  Math.copysign(pow, arg0) >>> gen_type(type)

                :expm1 ->
                  Math.expm1(arg0) >>> gen_type(type)

                :log1p ->
                  Math.log1p(arg0) >>> gen_type(type)

                :atan ->
                  Math.atan(arg0) >>> gen_type(type)
              end

            Linalg.yield(result) >>> []
          end
        end
      end >>> gen_type(t)
    end
  end

  # binary linalg
  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{type: type, data: %Nx.Defn.Expr{op: op, args: [a, b]}} = t
      )
      when op in [:remainder, :atan2, :power] do
    mlir block: block, ctx: ctx do
      a_value = gen_op(env, a)
      a_value = gen_expand(env, a_value, a, t)
      b_value = gen_op(env, b)
      b_value = gen_expand(env, b_value, b, t)

      out_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ODS.operand_segment_sizes([0, 0, 0])) >>>
          gen_type(t)

      Linalg.generic [
        a_value,
        b_value,
        out_tensor,
        operand_segment_sizes: ODS.operand_segment_sizes([2, 1]),
        indexing_maps: Manx.Linalg.gen_indexing_maps(a.shape, b.shape, t.shape),
        iterator_types: Manx.Linalg.gen_iterator_types(a.shape, b.shape, t.shape)
      ] do
        region do
          block bb0(arg0 >>> gen_type(type), arg1 >>> gen_type(type), out >>> gen_type(type)) do
            %MLIR.Value{} = out

            result =
              case op do
                :remainder ->
                  case type do
                    {:f, _} ->
                      Arith.remf(arg0, arg1) >>> gen_type(type)

                    {:i, _} ->
                      Arith.remui(arg0, arg1) >>> gen_type(type)

                    {:s, _} ->
                      Arith.remsi(arg0, arg1) >>> gen_type(type)
                  end

                :power ->
                  case type do
                    {:f, _} ->
                      Math.powf(arg0, arg1) >>> gen_type(type)

                    {inter_type, _} when inter_type in [:i, :s] ->
                      Math.ipowi(arg0, arg1) >>> gen_type(type)
                  end

                :atan2 ->
                  Math.atan2(arg0, arg1) >>> gen_type(type)
              end

            Linalg.yield(result) >>> []
          end
        end
      end >>> gen_type(t)
    end
  end

  def gen_op(env, %Nx.Tensor{
        data: %Nx.Defn.Expr{
          op: :optional,
          args:
            [
              %{
                data: %{op: :logical_not}
              },
              %{
                data: %{op: :equal}
              }
            ] = list
        }
      }) do
    gen_op(env, List.first(list))
  end

  # dot product
  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            op: :dot,
            args: [
              %Nx.Tensor{shape: {n}} = a,
              _,
              _,
              %Nx.Tensor{shape: {n}} = b,
              _,
              _
            ]
          }
        } = t
      ) do
    mlir block: block, ctx: ctx do
      a_value = gen_op(env, a)
      b_value = gen_op(env, b)
      a_value = TOSA.cast(a_value) >>> gen_type(%{a | type: t.type})
      b_value = TOSA.cast(b_value) >>> gen_type(%{b | type: t.type})
      c = TOSA.mul(a_value, b_value, shift: Attribute.integer(Type.i32(), 0)) >>> gen_type(a)

      c =
        TOSA.reduce_sum(c, axis: Attribute.integer(Type.i64(), 0)) >>> gen_type(%{t | shape: {1}})

      Tensor.collapse_shape(c, reassociation: Tensor.reassociation([])) >>> gen_type(t)
    end
  end

  # standard batch matmul
  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            op: :dot,
            args: [
              %Nx.Tensor{shape: a_shape} = a,
              [2],
              [0],
              %Nx.Tensor{shape: b_shape} = b,
              [1],
              [0]
            ]
          }
        } = t
      )
      when tuple_size(a_shape) == 3 and tuple_size(b_shape) == 3 do
    mlir block: block, ctx: ctx do
      a_value = gen_op(env, a)
      b_value = gen_op(env, b)

      TOSA.matmul(a_value, b_value) >>> gen_type(t)
    end
  end

  # generic dot product
  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            op: :dot,
            args:
              [
                %Nx.Tensor{shape: a_shape} = a,
                _contract_axes1,
                _batch_axes1,
                %Nx.Tensor{shape: b_shape} = b,
                _contract_axes2,
                _batch_axes2
              ] = args
          }
        } = t
      )
      when tuple_size(a_shape) in [2, 3] or tuple_size(b_shape) in [2, 3] do
    mlir block: block, ctx: ctx do
      a_value = gen_op(env, a)
      b_value = gen_op(env, b)
      a_value = TOSA.cast(a_value) >>> gen_type(%{a | type: t.type})
      b_value = TOSA.cast(b_value) >>> gen_type(%{b | type: t.type})
      {batched_a, batched_b} = Manx.Nx.Batcher.from_args(args)

      output_type = gen_type(t)

      out_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ODS.operand_segment_sizes([0, 0, 0])) >>>
          output_type

      zero =
        case t.type do
          {:f, _} ->
            Arith.constant(value: Attribute.float(gen_type(t.type), 0.0)) >>> gen_type(t.type)

          _ ->
            Arith.constant(value: Attribute.integer(gen_type(t.type), 0)) >>> gen_type(t.type)
        end

      out_tensor =
        Linalg.fill [zero, out_tensor, operand_segment_sizes: ODS.operand_segment_sizes([1, 1])] do
          region do
            block bb0(
                    arg >>> gen_type(t.type),
                    res >>> gen_type(t.type)
                  ) do
              %MLIR.Value{} = res
              Linalg.yield(arg) >>> []
            end
          end
        end >>> output_type

      Linalg.generic [
        a_value,
        b_value,
        out_tensor,
        operand_segment_sizes: ODS.operand_segment_sizes([2, 1]),
        indexing_maps: Manx.Nx.Batcher.gen_indexing_maps(batched_a, batched_b, t),
        iterator_types: Manx.Nx.Batcher.gen_iterator_types(batched_a, batched_b, t)
      ] do
        region do
          block bb0(
                  left >>> gen_type(t.type),
                  right >>> gen_type(t.type),
                  sum >>> gen_type(t.type)
                ) do
            sum =
              case t.type do
                {:f, _} ->
                  mul = Arith.mulf(left, right) >>> gen_type(t.type)
                  Arith.addf(sum, mul) >>> gen_type(t.type)

                _ ->
                  mul = Arith.muli(left, right) >>> gen_type(t.type)
                  Arith.addi(sum, mul) >>> gen_type(t.type)
              end

            Linalg.yield(sum) >>> []
          end
        end
      end >>> output_type
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            op: :concatenate,
            args: [
              inputs,
              axis
            ]
          }
        } = t
      )
      when is_list(inputs) do
    mlir block: block, ctx: ctx do
      inputs = inputs |> Enum.map(&gen_op(env, &1))
      TOSA.concat(inputs, axis: Attribute.integer(Type.i(64), axis)) >>> gen_type(t)
    end
  end

  # binary tosa
  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{data: %Nx.Defn.Expr{op: op, args: [%Nx.Tensor{} = a, %Nx.Tensor{} = b]}} = t
      ) do
    mlir block: block, ctx: ctx do
      a_t = %{a | type: t.type} |> gen_type
      b_t = %{b | type: t.type} |> gen_type
      a_value = gen_op(env, a)
      b_value = gen_op(env, b)

      {a_value, b_value} =
        case op do
          _ when op in [:equal, :greater_equal, :less_equal, :less, :greater, :not_equal] ->
            case {a.type, b.type} do
              {{int_type, _}, {:f, _}} when int_type in [:s, :u] ->
                a_value = TOSA.cast(a_value) >>> gen_type(%{a | type: b.type})
                {a_value, b_value}

              {{:f, _}, {int_type, _}} when int_type in [:s, :u] ->
                b_value = TOSA.cast(b_value) >>> gen_type(%{b | type: a.type})
                {a_value, b_value}

              {{_, width_a}, {_, width_b}}
              when width_a > width_b ->
                b_value = TOSA.cast(b_value) >>> gen_type(%{b | type: a.type})
                {a_value, b_value}

              {{_, width_a}, {_, width_b}}
              when width_a < width_b ->
                a_value = TOSA.cast(a_value) >>> gen_type(%{a | type: b.type})
                {a_value, b_value}

              _ ->
                b_value = TOSA.cast(b_value) >>> gen_type(%{b | type: a.type})
                {a_value, b_value}
            end

          _ when op in [:logical_or, :logical_xor, :logical_and] ->
            a_value = TOSA.cast(a_value) >>> gen_type(%{a | type: {:u, 1}})
            b_value = TOSA.cast(b_value) >>> gen_type(%{b | type: {:u, 1}})
            {a_value, b_value}

          _ ->
            a_value = TOSA.cast(a_value) >>> a_t
            b_value = TOSA.cast(b_value) >>> b_t
            {a_value, b_value}
        end

      case op do
        :subtract ->
          TOSA.sub(a_value, b_value) >>> gen_type(t)

        :less_equal ->
          c = TOSA.greater_equal(b_value, a_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :greater_equal ->
          c = TOSA.greater_equal(a_value, b_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :less ->
          c = TOSA.greater(b_value, a_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :greater ->
          c = TOSA.greater(a_value, b_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :equal ->
          c = TOSA.equal(b_value, a_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :not_equal ->
          c = TOSA.equal(b_value, a_value) >>> gen_type(%{t | type: {:u, 1}})
          c = TOSA.logical_not(c) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :logical_and ->
          c = TOSA.logical_and(a_value, b_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :logical_or ->
          c = TOSA.logical_or(a_value, b_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :logical_xor ->
          c = TOSA.logical_xor(a_value, b_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :add ->
          TOSA.add(a_value, b_value) >>> gen_type(t)

        :max ->
          TOSA.maximum(a_value, b_value) >>> gen_type(t)

        :min ->
          TOSA.minimum(a_value, b_value) >>> gen_type(t)

        :bitwise_and ->
          TOSA.bitwise_and(a_value, b_value) >>> gen_type(t)

        :bitwise_or ->
          TOSA.bitwise_or(a_value, b_value) >>> gen_type(t)

        :bitwise_xor ->
          TOSA.bitwise_xor(a_value, b_value) >>> gen_type(t)

        :left_shift ->
          TOSA.logical_left_shift(a_value, b_value) >>> gen_type(t)

        :right_shift ->
          case t.type do
            {:u, _} ->
              TOSA.logical_right_shift(a_value, b_value) >>> gen_type(t)

            {:s, _} ->
              TOSA.arithmetic_right_shift(a_value, b_value, round: Attribute.bool(false)) >>>
                gen_type(t)
          end

        :multiply ->
          TOSA.mul(a_value, b_value, shift: Attribute.integer(Type.i32(), 0)) >>> gen_type(t)

        :divide ->
          b_r = TOSA.reciprocal(b_value) >>> b_t
          TOSA.mul(a_value, b_r, shift: Attribute.integer(Type.i(32), 0)) >>> gen_type(t)

        :quotient ->
          a_value = TOSA.cast(a_value) >>> gen_type(%{a | type: {:u, 32}})
          b_value = TOSA.cast(b_value) >>> gen_type(%{b | type: {:u, 32}})
          result = TOSA.div(a_value, b_value) >>> gen_type(%{t | type: {:u, 32}})
          TOSA.cast(result) >>> gen_type(t)

        _ ->
          raise "Unsupported binary op: #{inspect(t, structs: false, pretty: true)}"
      end
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{data: %Nx.Defn.Expr{op: :select, args: [pred, on_true, on_false]}} = t
      ) do
    mlir block: block, ctx: ctx do
      pred_value = gen_op(env, pred)
      pred_t = %{pred | type: {:u, 1}}
      pred_value = TOSA.cast(pred_value) >>> gen_type(pred_t)
      pred_value = gen_expand(env, pred_value, pred_t, t)
      on_true_value = gen_op(env, on_true)
      on_false_value = gen_op(env, on_false)
      on_true_value = TOSA.cast(on_true_value) >>> gen_type(%{on_true | type: t.type})
      on_false_value = TOSA.cast(on_false_value) >>> gen_type(%{on_false | type: t.type})
      TOSA.select(pred_value, on_true_value, on_false_value) >>> gen_type(t)
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{data: %Nx.Defn.Expr{op: :reshape, args: [input]}} = t
      ) do
    mlir block: block, ctx: ctx do
      input = gen_op(env, input)

      new_shape =
        t.shape
        |> Tuple.to_list()
        |> Attribute.dense_array(Beaver.Native.I64)

      TOSA.reshape(input, new_shape: new_shape) >>> gen_type(t)
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{data: %Nx.Defn.Expr{op: :as_type, args: [input1]}} = t
      ) do
    mlir block: block, ctx: ctx do
      input1_value = gen_op(env, input1)
      TOSA.cast(input1_value) >>> gen_type(t)
    end
  end

  def gen_op(
        %Env{block: block, ctx: ctx} = env,
        %Nx.Tensor{data: %Nx.Defn.Expr{op: :iota, args: [axis]} = expr, shape: out_shape} = t
      ) do
    if axis do
      mlir block: block, ctx: ctx do
        out_tensor =
          Bufferization.alloc_tensor(operand_segment_sizes: ODS.operand_segment_sizes([0, 0, 0])) >>>
            gen_type(t)

        Linalg.generic [
          out_tensor,
          operand_segment_sizes: ODS.operand_segment_sizes([0, 1]),
          indexing_maps: Manx.Linalg.gen_indexing_maps(out_shape),
          iterator_types: Manx.Linalg.gen_iterator_types(out_shape)
        ] do
          region do
            block bb0(arg1 >>> gen_type(t.type)) do
              %MLIR.Value{} = arg1
              index = Linalg.index(dim: Attribute.integer(Type.i64(), axis)) >>> Type.index()
              cast = Arith.index_cast(index) >>> gen_type(t.type)
              Linalg.yield(cast) >>> []
            end
          end
        end >>> gen_type(t)
      end
    else
      mlir block: block, ctx: ctx do
        dim = t.shape |> Tuple.to_list() |> Enum.reduce(1, &Kernel.*/2)

        permutation_1d =
          gen_op(
            env,
            %{t | data: %{expr | args: [0]}, shape: {dim}}
          )

        new_shape =
          t.shape
          |> Tuple.to_list()
          |> Attribute.dense_array(Beaver.Native.I64)

        # this should generate affine.apply for index
        TOSA.reshape(permutation_1d, new_shape: new_shape) >>> gen_type(t)
      end
    end
  end

  def gen_op(%Env{} = env, tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&gen_op(env, &1))
    |> List.to_tuple()
  end

  def gen_op(_, tensor) do
    raise "op not supported: " <> inspect(tensor)
  end
end
