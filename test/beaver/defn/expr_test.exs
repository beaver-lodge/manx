defmodule Manx.ExprTest do
  use ExUnit.Case, async: true
  import Nx.Defn
  import Manx.Assert

  @moduletag :nx
  setup do
    Nx.Defn.default_options(compiler: Manx.Compiler)
    :ok
  end

  describe "tuples" do
    defn add_subtract_tuple(a, b), do: {a + b, a - b}

    test "on results" do
      assert_equal(add_subtract_tuple(2, 3), {Nx.tensor(5), Nx.tensor(-1)})

      assert_equal(
        add_subtract_tuple(Nx.tensor([-1, 0, 1]), Nx.tensor([10, 10, 10])),
        {Nx.tensor([9, 10, 11]), Nx.tensor([-11, -10, -9])}
      )

      assert_equal(
        add_subtract_tuple(Nx.tensor([-1, 0, 1]), 10),
        {Nx.tensor([9, 10, 11]), Nx.tensor([-11, -10, -9])}
      )
    end

    defn pattern_tuple({a, b}), do: a + b

    test "on patterns" do
      assert_equal(pattern_tuple({2, 3}), Nx.tensor(5))

      assert_equal(
        pattern_tuple({Nx.tensor([1, 2]), Nx.tensor([[3], [4]])}),
        Nx.tensor([[4, 5], [5, 6]])
      )
    end

    defn calls_pattern_tuple(a, b), do: pattern_tuple({a, b})

    test "on inlined tuples" do
      assert_equal(calls_pattern_tuple(2, 3), Nx.tensor(5))

      assert_equal(
        calls_pattern_tuple(Nx.tensor([1, 2]), Nx.tensor([[3], [4]])),
        Nx.tensor([[4, 5], [5, 6]])
      )
    end
  end

  describe "tensor constants" do
    @two 2
    defn constants, do: @two
    defn add_two_attribute(t), do: t + @two

    @two_per_two Nx.tensor([[1, 2], [3, 4]])
    defn add_2x2_attribute(t), do: t + @two_per_two
    defn add_2x2_constant(), do: @two_per_two + @two_per_two
    defn add_2x2_constant(_), do: @two_per_two + @two_per_two

    test "handles tensors as constants" do
      assert_equal(constants(), Nx.tensor(2))
    end

    test "expands module attributes to scalars" do
      assert_equal(add_two_attribute(1), Nx.tensor(3))
      assert_equal(add_two_attribute(Nx.tensor([1, 2, 3])), Nx.tensor([3, 4, 5]))
    end

    test "expands module attributes to tensors" do
      assert_equal(add_2x2_attribute(1), Nx.tensor([[2, 3], [4, 5]]))
      assert_equal(add_2x2_attribute(Nx.tensor([1, 2])), Nx.tensor([[2, 4], [4, 6]]))
    end

    test "constants should be folded" do
      assert_equal(add_2x2_constant(), Nx.tensor([[2, 4], [6, 8]]))
      assert_equal(add_2x2_constant(1), Nx.tensor([[2, 4], [6, 8]]))
    end
  end

  describe "non finite" do
    defn infinity, do: Nx.Constants.infinity()
    defn neg_infinity, do: Nx.Constants.neg_infinity()
    defn nan, do: Nx.Constants.nan()

    test "handles non-finite constants correctly" do
      assert_equal(infinity(), Nx.Constants.infinity())
      assert_equal(neg_infinity(), Nx.Constants.neg_infinity())
      # TODO: fix this
      {nan(), Nx.Constants.nan()}
    end

    defn negate_infinity, do: Nx.negate(Nx.Constants.infinity())
    defn negate_neg_infinity, do: Nx.negate(Nx.Constants.infinity())

    test "sanity check constants" do
      assert_equal(negate_infinity(), Nx.Constants.neg_infinity())
      assert_equal(infinity(), Nx.Constants.infinity())
    end
  end

  describe "float16" do
    defn return_float, do: Nx.tensor(1, type: {:f, 16})

    test "supports float16 return types" do
      assert_equal(return_float(), Nx.tensor(1, type: {:f, 16}))
    end
  end

  describe "complex" do
    defn return_complex, do: Nx.complex(1, 2)
    defn return_complex_tensor, do: Nx.broadcast(Nx.complex(1, 2), {3, 3, 3})

    test "supports complex return types" do
      assert_equal(return_complex(), Nx.tensor(Complex.new(1, 2)))
      assert_equal(return_complex_tensor(), Nx.broadcast(Complex.new(1, 2), {3, 3, 3}))
    end
  end

  describe "conjugate" do
    defn conjugate(x), do: Nx.conjugate(x)

    test "correctly returns complex conjugate" do
      assert_equal(conjugate(Nx.tensor(Complex.new(1, 2))), Nx.tensor(Complex.new(1, -2)))
      # This differs from the Nx doctest, which I believe should also return -0
      assert_equal(conjugate(Nx.tensor(1)), Nx.tensor(Complex.new(1, -0.0)))

      assert_equal(
        conjugate(Nx.tensor([Complex.new(1, 2), Complex.new(2, -4)])),
        Nx.tensor([Complex.new(1, -2), Complex.new(2, 4)])
      )
    end
  end

  describe "imag" do
    defn imag(x), do: Nx.imag(x)

    test "correctly returns imaginary part of complex" do
      assert_equal(imag(Nx.tensor(Complex.new(1, 2))), Nx.tensor(2.0))
      assert_equal(imag(Nx.tensor(1)), Nx.tensor(0.0))

      assert_equal(
        imag(Nx.tensor([Complex.new(1, 2), Complex.new(2, -4)])),
        Nx.tensor([2.0, -4.0])
      )
    end
  end

  describe "+/2" do
    @describetag :plus
    defn add_two(a, b), do: a + b

    test "same shape and type" do
      assert_equal(add_two(1.0, 2.0), Nx.tensor(3.0))
      assert_equal(add_two(1, 2), Nx.tensor(3))

      assert_equal(add_two(Nx.tensor([1, 2]), Nx.tensor([3, 4])), Nx.tensor([4, 6]))
      assert_equal(add_two(Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])), Nx.tensor([4.0, 6.0]))
    end

    test "different types" do
      tensors = [
        {1, 2},
        {1.0, 2},
        {1.0, 3.0},
        {Nx.tensor([1, 2], type: {:u, 8}), 3},
        {Nx.tensor([1, 2], type: {:u, 8}), -3},
        {Nx.tensor([1, 2], type: {:u, 8}), 3.0},
        {Nx.tensor([1, 2], type: {:s, 8}), 3},
        {Nx.tensor([1, 2], type: {:s, 8}), 3.0},
        {Nx.tensor([1, 2], type: {:f, 32}), 3},
        {Nx.tensor([1, 2], type: {:f, 32}), 3.0},
        {Nx.tensor([1, 2], type: {:u, 8}), Nx.tensor(3, type: {:u, 16})},
        {Nx.tensor([1, 2], type: {:u, 8}), Nx.tensor(-3, type: {:s, 16})},
        {Nx.tensor([1, 2], type: {:u, 8}), Nx.tensor(3.0, type: {:f, 32})},
        {Nx.tensor([1, 2], type: {:s, 8}), Nx.tensor(3, type: {:s, 16})},
        {Nx.tensor([1, 2], type: {:s, 8}), Nx.tensor(3.0, type: {:f, 32})},
        {Nx.tensor([1, 2], type: {:f, 32}), Nx.tensor(3, type: {:u, 16})},
        {Nx.tensor([1, 2], type: {:f, 32}), Nx.tensor(3, type: {:s, 16})}
        # {Nx.tensor([1, 2], type: {:f, 32}), Nx.tensor(3.0, type: {:f, 64})}
      ]

      for {left, right} <- tensors do
        assert_all_close(add_two(left, right), evaluate(&add_two/2, [left, right]))
        assert_all_close(add_two(right, left), evaluate(&add_two/2, [right, left]))
      end
    end

    defn add_two_int(t), do: t + 2
    defn add_two_float(t), do: t + 2.0

    test "constants" do
      tensors = [
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:u, 16}),
        Nx.tensor([1, 2], type: {:u, 32}),
        Nx.tensor([1, 2], type: {:s, 8}),
        Nx.tensor([1, 2], type: {:s, 32}),
        Nx.tensor([1, 2], type: {:f, 32})
        # Nx.tensor([1, 2], type: {:f, 64})
      ]

      for t <- tensors do
        assert_equal(add_two_int(t), Nx.add(t, 2))
        assert_equal(add_two_float(t), Nx.add(t, 2.0))
      end
    end

    test "broadcast" do
      tensors = [
        {Nx.tensor([1, 2]), Nx.tensor([[1, 2], [3, 4]])},
        {Nx.tensor([1, 2]), Nx.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]])},
        {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
        {Nx.tensor([[10, 20]]), Nx.tensor([[1], [2]])},
        {Nx.tensor([[[10], [20]]]), Nx.tensor([[[1, 2]], [[3, 4]]])},
        {Nx.tensor([[[100], [200], [300]]]),
         Nx.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]])},
        {Nx.tensor([[[[1]]]]), Nx.tensor([[1, 2], [3, 4]])},
        {Nx.tensor([[[[1]]]]), Nx.tensor([1, 2])},
        {Nx.tensor([[[10], [20]], [[30], [40]]]), Nx.tensor([[1, 2]])},
        {Nx.tensor([[[[10], [20]], [[30], [40]]]]), Nx.tensor([[[1, 2]], [[3, 4]]])},
        {Nx.tensor([[[[10], [20]], [[30], [40]]]]), Nx.tensor([[[[1, 2]]], [[[3, 4]]]])},
        {Nx.tensor([[[10], [20]], [[30], [40]]]), Nx.tensor([[[1, 2]], [[3, 4]]])}
      ]

      for {left, right} <- tensors do
        assert_all_close(add_two(left, right), evaluate(&add_two/2, [left, right]))
        assert_all_close(add_two(right, left), evaluate(&add_two/2, [right, left]))
      end
    end

    test "names" do
      left = Nx.tensor([[10, 20]], names: [nil, :tens])
      right = Nx.tensor([[1], [2]], names: [:ones, nil])
      assert add_two(left, right).names == [:ones, :tens]
    end
  end

  describe "//2" do
    defn divide_two(a, b), do: a / b

    test "parameters" do
      tensors = [
        {1, 2},
        {1, Nx.tensor([1.0, 2.0, 3.0])},
        {Nx.tensor([1, 2, 3]), 1},
        {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
        {Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8})},
        {Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32})}
      ]

      for {left, right} <- tensors do
        assert_all_close(divide_two(left, right), Nx.divide(left, right))
        assert_all_close(divide_two(right, left), Nx.divide(right, left))
      end
    end

    defn divide_two_int(t), do: t / 2
    defn divide_two_float(t), do: t / 2.0

    test "constants" do
      tensors = [
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:u, 16}),
        Nx.tensor([1, 2], type: {:u, 32}),
        Nx.tensor([1, 2], type: {:s, 8}),
        Nx.tensor([1, 2], type: {:s, 32}),
        Nx.tensor([1, 2], type: {:f, 32})
        # Nx.tensor([1, 2], type: {:f, 64})
      ]

      for t <- tensors do
        assert_all_close(divide_two_int(t), Nx.divide(t, 2))
        assert_all_close(divide_two_float(t), Nx.divide(t, 2.0))
      end
    end
  end

  describe "remainder" do
    defn remainder(a, b), do: Nx.remainder(a, b)

    test "integers" do
      left = Nx.tensor([-1023, 1023])
      right = Nx.tensor([[-4], [4]])
      assert Nx.shape(remainder(left, right)) == {2, 2}
      assert_all_close(remainder(left, right), Nx.remainder(left, right))
    end

    test "floats" do
      left = Nx.tensor([-8.3, -8.4, -8.5, 8.3, 8.4, 8.5])
      right = Nx.tensor([[-4.2], [-4.1], [-4.0], [4.0], [4.1], [4.2]])
      assert Nx.shape(remainder(left, right)) == {6, 6}
      assert_all_close(remainder(left, right), Nx.remainder(left, right))
    end
  end

  describe "element-wise arith operators" do
    @tensors [
      {1, 2},
      {1, Nx.tensor([1.0, 2.0, 3.0])},
      {Nx.tensor([1, 2, 3]), 1},
      {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
      {Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8})},
      {Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32})}
    ]

    defn subtract_two(a, b), do: a - b

    test "-" do
      for {left, right} <- @tensors do
        assert_all_close(subtract_two(left, right), Nx.subtract(left, right))
        assert_all_close(subtract_two(right, left), Nx.subtract(right, left))
      end
    end

    defn multiply_two(a, b), do: a * b

    test "*" do
      for {left, right} <- @tensors do
        assert_all_close(multiply_two(left, right), Nx.multiply(left, right))
        assert_all_close(multiply_two(right, left), Nx.multiply(right, left))
      end
    end

    defn unary_minus(a), do: -a

    test "negate" do
      for t <- [
            Nx.tensor([-1, 0, 1], type: {:u, 8}),
            Nx.tensor([-1, 0, 1]),
            Nx.tensor([-1.0, 1.0])
          ] do
        assert_equal(unary_minus(t), Nx.negate(t))
      end
    end

    defn max_two(a, b), do: max(a, b)

    test "max" do
      for {left, right} <- @tensors do
        assert_all_close(max_two(left, right), Nx.max(left, right))
        assert_all_close(max_two(right, left), Nx.max(right, left))
      end
    end

    defn min_two(a, b), do: min(a, b)

    test "min" do
      for {left, right} <- @tensors do
        assert_all_close(min_two(left, right), Nx.min(left, right))
        assert_all_close(min_two(right, left), Nx.min(right, left))
      end
    end

    defn power_two(a, b), do: Nx.power(a, b)

    test "power" do
      for {left, right} <- @tensors do
        case left do
          %{type: {_, 8}} ->
            nil

          i when is_integer(i) ->
            nil

          %{type: {:s, _}} ->
            nil

          _ ->
            assert_all_close(power_two(left, right), Nx.power(left, right))
            assert_all_close(power_two(right, left), Nx.power(right, left))
        end
      end
    end

    defn atan2_two(a, b), do: Nx.atan2(a, b)

    test "atan2" do
      <<neg_zero::float>> = <<0x8000000000000000::64>>
      left = Nx.tensor([-1.0, neg_zero, 0.0, 1.0])
      right = Nx.tensor([[-1.0], [neg_zero], [0.0], [1.0]])

      assert_all_close(atan2_two(left, right), Nx.atan2(left, right))
      assert_all_close(atan2_two(right, left), Nx.atan2(right, left))
    end

    defn quotient_two(a, b), do: Nx.quotient(a, b)

    test "quotient" do
      int_tensors = [
        {1, 2},
        {1, Nx.tensor([1, 2, 3])},
        {Nx.tensor([1, 2, 3]), 1},
        {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
        {Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8})},
        {Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 32})}
      ]

      for {left, right} <- int_tensors do
        assert_all_close(quotient_two(left, right), Nx.quotient(left, right))
        assert_all_close(quotient_two(right, left), Nx.quotient(right, left))
      end
    end
  end

  describe "element-wise bitwise operators" do
    @left Nx.tensor([-2, -1, 0, 1, 2])
    @right Nx.tensor([[-2], [-1], [0], [1], [2]])

    defn bitwise_and(a, b), do: a &&& b

    test "bitwise_and" do
      assert Nx.shape(bitwise_and(@left, @right)) == {5, 5}
      assert_equal(bitwise_and(@left, @right), Nx.bitwise_and(@left, @right))
    end

    defn bitwise_or(a, b), do: a ||| b

    test "bitwise_or" do
      assert Nx.shape(bitwise_or(@left, @right)) == {5, 5}
      assert_equal(bitwise_or(@left, @right), Nx.bitwise_or(@left, @right))
    end

    defn bitwise_not(a), do: ~~~a

    test "bitwise_not" do
      assert Nx.shape(bitwise_not(@left)) == {5}
      assert_equal(bitwise_not(@left), Nx.bitwise_not(@left))
    end

    defn bitwise_pc(a), do: Nx.population_count(a)

    test "population_count" do
      assert Nx.shape(bitwise_pc(@left)) == {5}
      assert_equal(bitwise_pc(@left), Nx.population_count(@left))
    end

    defn bitwise_clz(a), do: Nx.count_leading_zeros(a)

    test "count_leading_zeros" do
      assert Nx.shape(bitwise_clz(@left)) == {5}
      assert_equal(bitwise_clz(@left), Nx.count_leading_zeros(@left))
    end

    @left Nx.tensor([-2, -1, 0, 1, 2])
    @right Nx.tensor([[0], [1], [2], [3], [4]])

    defn left_shift(a, b), do: a <<< b

    test "left_shift" do
      assert Nx.shape(left_shift(@left, @right)) == {5, 5}
      assert_equal(left_shift(@left, @right), Nx.left_shift(@left, @right))
    end

    @left_signed Nx.tensor([-128, -127, -2, -1, 0, 1, 2, 126, 127], type: {:s, 8})
    @right_signed Nx.tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8]], type: {:s, 8})

    @left_unsigned Nx.tensor([0, 1, 2, 253, 254, 255], type: {:u, 8})
    @right_unsigned Nx.tensor([[0], [1], [2], [3], [4], [5]], type: {:u, 8})

    defn right_shift(a, b), do: a >>> b

    test "right_shift" do
      assert Nx.shape(right_shift(@left_signed, @right_signed)) == {9, 9}

      assert_equal(
        right_shift(@left_signed, @right_signed),
        Nx.right_shift(@left_signed, @right_signed)
      )

      assert Nx.shape(right_shift(@left_unsigned, @right_unsigned)) == {6, 6}

      assert_equal(
        right_shift(@left_unsigned, @right_unsigned),
        Nx.right_shift(@left_unsigned, @right_unsigned)
      )
    end
  end

  describe "exp" do
    defn exp(t), do: Nx.exp(t)

    test "computes the exp across types" do
      assert_all_close(
        Nx.tensor([1, 2, 3]) |> exp(),
        Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668])
      )

      assert_all_close(
        Nx.tensor([1, 2, 3], type: {:s, 8}) |> exp(),
        Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668], type: {:f, 32})
      )

      assert_all_close(
        Nx.tensor([1, 2, 3], type: {:u, 8}) |> exp(),
        Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668], type: {:f, 32})
      )

      assert_all_close(
        Nx.tensor([1.0, 2.0, 3.0]) |> exp(),
        Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668])
      )

      assert_all_close(
        Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}) |> exp(),
        Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668], type: {:f, 32})
      )
    end
  end

  describe "equal" do
    defn equal(a, b), do: Nx.equal(a, b)

    test "computes equality of scalars" do
      assert_equal(equal(Nx.tensor(1), Nx.tensor(2)), Nx.tensor(0, type: {:u, 8}))
    end

    test "computes equality with broadcasting" do
      assert_equal(
        equal(Nx.tensor(1), Nx.tensor([1, 2, 3])),
        Nx.tensor([1, 0, 0], type: {:u, 8})
      )
    end

    test "computes equality with mixed types" do
      assert_equal(
        equal(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])),
        Nx.tensor([1, 1, 1], type: {:u, 8})
      )
    end

    defn successive_compare(y_true, y_pred) do
      y_pred
      |> Nx.equal(y_pred)
      |> Nx.equal(y_true)
    end

    @tag :todo
    # TODO: track https://github.com/llvm/llvm-project/issues/57951
    test "computes successive comparisons" do
      {successive_compare(Nx.tensor(1), Nx.tensor(1)), Nx.tensor(1, type: {:u, 8})}
    end
  end

  describe "not equal" do
    defn not_equal(a, b), do: Nx.not_equal(a, b)

    test "computes equality of scalars" do
      assert_equal(not_equal(Nx.tensor(1), Nx.tensor(2)), Nx.tensor(1, type: {:u, 8}))
    end

    test "computes equality with broadcasting" do
      assert_equal(
        not_equal(Nx.tensor(1), Nx.tensor([1, 2, 3])),
        Nx.tensor([0, 1, 1], type: {:u, 8})
      )
    end

    test "computes equality with mixed types" do
      assert_equal(
        not_equal(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])),
        Nx.tensor([0, 0, 0], type: {:u, 8})
      )
    end
  end

  describe "less" do
    defn less(a, b), do: Nx.less(a, b)

    test "compares scalars" do
      assert_equal(less(Nx.tensor(1), Nx.tensor(2)), Nx.tensor(1, type: {:u, 8}))
    end

    test "compares with broadcasting" do
      assert_equal(less(Nx.tensor(1), Nx.tensor([1, 2, 3])), Nx.tensor([0, 1, 1], type: {:u, 8}))
    end

    test "compares with mixed types" do
      assert_equal(
        less(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])),
        Nx.tensor([0, 0, 0], type: {:u, 8})
      )
    end
  end

  describe "greater" do
    defn greater(a, b), do: Nx.greater(a, b)

    test "compares scalars" do
      assert_equal(greater(Nx.tensor(1), Nx.tensor(2)), Nx.tensor(0, type: {:u, 8}))
    end

    test "compares with broadcasting" do
      assert_equal(
        greater(Nx.tensor(1), Nx.tensor([1, 2, 3])),
        Nx.tensor([0, 0, 0], type: {:u, 8})
      )
    end

    test "compares with mixed types" do
      assert_equal(
        greater(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])),
        Nx.tensor([0, 0, 0], type: {:u, 8})
      )
    end
  end

  describe "less equal" do
    defn less_equal(a, b), do: Nx.less_equal(a, b)

    test "compares scalars" do
      assert_equal(less_equal(Nx.tensor(1), Nx.tensor(2)), Nx.tensor(1, type: {:u, 8}))
    end

    test "compares with broadcasting" do
      assert_equal(
        less_equal(Nx.tensor(1), Nx.tensor([1, 2, 3])),
        Nx.tensor([1, 1, 1], type: {:u, 8})
      )
    end

    test "compares with mixed types" do
      assert_equal(
        less_equal(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])),
        Nx.tensor([1, 1, 1], type: {:u, 8})
      )
    end
  end

  describe "greater equal" do
    defn greater_equal(a, b), do: Nx.greater_equal(a, b)

    test "compares scalars" do
      assert_equal(greater_equal(Nx.tensor(1), Nx.tensor(2)), Nx.tensor(0, type: {:u, 8}))
    end

    test "compares with broadcasting" do
      assert_equal(
        greater_equal(Nx.tensor(1), Nx.tensor([1, 2, 3])),
        Nx.tensor([1, 0, 0], type: {:u, 8})
      )
    end

    test "compares with mixed types" do
      assert_equal(
        greater_equal(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])),
        Nx.tensor([1, 1, 1], type: {:u, 8})
      )
    end
  end

  describe "logical" do
    defn logical_and(a, b), do: Nx.logical_and(a, b)

    test "and" do
      assert_equal(
        logical_and(Nx.tensor([-1, 0, 1]), Nx.tensor([[-1], [0], [1]])),
        Nx.tensor(
          [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
          ],
          type: {:u, 8}
        )
      )

      assert_equal(
        logical_and(Nx.tensor([-1.0, 0.0, 1.0]), Nx.tensor([[-1], [0], [1]])),
        Nx.tensor(
          [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
          ],
          type: {:u, 8}
        )
      )
    end

    defn logical_or(a, b), do: Nx.logical_or(a, b)

    test "or" do
      assert_equal(
        logical_or(Nx.tensor([-1, 0, 1]), Nx.tensor([[-1], [0], [1]])),
        Nx.tensor(
          [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
          ],
          type: {:u, 8}
        )
      )

      assert_equal(
        logical_or(Nx.tensor([-1.0, 0.0, 1.0]), Nx.tensor([[-1], [0], [1]])),
        Nx.tensor(
          [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
          ],
          type: {:u, 8}
        )
      )
    end

    defn logical_xor(a, b), do: Nx.logical_xor(a, b)

    test "xor" do
      assert_equal(
        logical_xor(Nx.tensor([-1, 0, 1]), Nx.tensor([[-1], [0], [1]])),
        Nx.tensor(
          [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
          ],
          type: {:u, 8}
        )
      )

      assert_equal(
        logical_xor(Nx.tensor([-1.0, 0.0, 1.0]), Nx.tensor([[-1], [0], [1]])),
        Nx.tensor(
          [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
          ],
          type: {:u, 8}
        )
      )
    end

    defn logical_not(a), do: Nx.logical_not(a)

    test "not" do
      assert_equal(
        logical_not(Nx.tensor([-2, -1, 0, 1, 2])),
        Nx.tensor([0, 0, 1, 0, 0], type: {:u, 8})
      )
    end
  end

  describe "select" do
    defn select(pred, x, y), do: Nx.select(pred, x, y)

    test "selects one or the other with a scalar" do
      assert_equal(
        select(Nx.tensor(1), Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])),
        Nx.tensor([1, 2, 3])
      )
    end

    test "selects with type" do
      assert_equal(
        select(
          Nx.tensor(1),
          Nx.tensor([1, 2, 3], type: {:u, 8}),
          Nx.tensor([4, 5, 6], type: {:u, 8})
        ),
        Nx.tensor([1, 2, 3], type: {:u, 8})
      )

      assert_equal(
        select(
          Nx.tensor(1),
          Nx.tensor([1, 2, 3], type: {:u, 8}),
          Nx.tensor([4, 5, 6], type: {:f, 32})
        ),
        Nx.tensor([1, 2, 3], type: {:f, 32})
      )
    end

    test "selects with broadcasting" do
      assert_equal(
        select(Nx.tensor([1, 0, 1, 0, 1]), Nx.tensor([10]), Nx.tensor([1, 2, 3, 4, 5])),
        Nx.tensor([10, 2, 10, 4, 10])
      )

      assert_equal(
        select(Nx.tensor([-2, -1, 0, 1, 2]), Nx.tensor([10]), Nx.tensor([1, 2, 3, 4, 5])),
        Nx.tensor([10, 10, 3, 10, 10])
      )
    end
  end

  describe "unary float ops" do
    @int_tensor Nx.tensor([1, 2, 3])
    @float_tensor Nx.tensor([1.0, 2.0, 3.0])
    float_ops =
      ([
         :exp,
         :expm1,
         :log,
         :log1p,
         :sigmoid,
         :cos,
         :sin,
         :tanh,
         :sqrt,
         :rsqrt,
         :cbrt,
         :is_nan
       ] ++
         [:is_infinity, :tan, :acosh, :asinh, :cosh, :sinh, :erf, :erfc])
      |> Enum.reject(fn x -> x in [:erfc, :asinh, :sinh, :acosh, :cosh] end)

    for fun <- float_ops do
      defn_fun = :"unary_#{fun}"
      defn_var = Macro.var(defn_fun, __MODULE__)
      defn unquote(defn_fun)(t), do: Nx.unquote(fun)(t)

      test "#{fun}" do
        assert_all_close(
          unquote(defn_fun)(@float_tensor),
          evaluate(&(unquote(defn_var) / 1), [@float_tensor])
        )

        assert_all_close(
          unquote(defn_fun)(@int_tensor),
          evaluate(&(unquote(defn_var) / 1), [@int_tensor])
        )
      end
    end
  end

  describe "softmax" do
    defn softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t))

    test "computes softmax" do
      assert_all_close(
        softmax(Nx.tensor([1.0, 2.0, 3.0, 4.0])),
        Nx.tensor([
          0.03205860328008499,
          0.08714431874203257,
          0.23688281808991013,
          0.6439142598879722
        ])
      )
    end
  end

  describe "dot product" do
    defn dot(a, b), do: Nx.dot(a, b)

    test "computes the dot product of scalars" do
      assert_equal(dot(Nx.tensor(2), Nx.tensor(2)), Nx.tensor(4))
      assert_equal(dot(Nx.tensor(2.0), Nx.tensor(2.0)), Nx.tensor(4.0))
      assert_equal(dot(Nx.tensor(-2.0), Nx.tensor(-2)), Nx.tensor(4.0))
    end

    test "computes the dot product of vectors" do
      assert_equal(
        dot(Nx.tensor([1, 2, 3], type: {:s, 32}), Nx.tensor([4, 5, 6], type: {:s, 32})),
        Nx.tensor(32, type: {:s, 32})
      )

      assert_equal(
        dot(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}), Nx.tensor([4, 5, 6])),
        Nx.tensor(32.0)
      )

      assert_equal(dot(Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([4.0, 5.0, 6.0])), Nx.tensor(32.0))
    end

    test "computes the dot product of matrices" do
      assert_equal(
        dot(
          Nx.tensor([[1, 2, 3], [4, 5, 6]], type: {:s, 32}),
          Nx.tensor([[7, 8], [9, 10], [11, 12]], type: {:s, 32})
        ),
        Nx.tensor([[58, 64], [139, 154]], type: {:s, 32})
      )

      assert_equal(
        dot(
          Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          Nx.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        ),
        Nx.tensor([[58.0, 64.0], [139.0, 154.0]])
      )

      assert_equal(
        dot(
          Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          Nx.tensor([[7, 8], [9, 10], [11, 12]])
        ),
        Nx.tensor([[58.0, 64.0], [139.0, 154.0]])
      )
    end

    test "computes the dot product of tensors" do
      assert_equal(
        dot(
          Nx.tensor(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
            type: {:s, 32},
            names: [:a, :b, :c]
          ),
          Nx.tensor(
            [[[1, 2, 3], [3, 4, 5], [5, 6, 7]]],
            type: {:s, 32},
            names: [:e, :f, :g]
          )
        ),
        Nx.tensor(
          [
            [[[22, 28, 34]], [[49, 64, 79]], [[76, 100, 124]]],
            [[[22, 28, 34]], [[49, 64, 79]], [[76, 100, 124]]]
          ],
          type: {:s, 32},
          names: [:a, :b, :e, :g]
        )
      )
    end

    defn batched_dot(t1, t2), do: Nx.dot(t1, [1], [0], t2, [1], [0])

    test "computes a batched dot product" do
      assert_equal(
        batched_dot(Nx.iota({3, 2, 3}, type: {:f, 32}), Nx.iota({3, 2, 2}, type: {:f, 32})),
        Nx.tensor([
          [[6.0, 9.0], [8.0, 13.0], [10.0, 17.0]],
          [[78.0, 93.0], [88.0, 105.0], [98.0, 117.0]],
          [[246.0, 273.0], [264.0, 293.0], [282.0, 313.0]]
        ])
      )
    end

    defn general_dot(t1, t2), do: Nx.dot(t1, [0, 1], [], t2, [1, 2], [])

    test "computes a general dot product" do
      assert_equal(
        general_dot(Nx.iota({4, 5, 2}, type: {:f, 32}), Nx.iota({2, 4, 5}, type: {:f, 32})),
        Nx.tensor([[4940.0, 12540.0], [5130.0, 13130.0]])
      )
    end
  end

  describe "creation" do
    defn iota, do: Nx.iota({10, 10})

    test "iota" do
      %{data: %Manx{}} = tensor = iota()
      assert Nx.backend_transfer(tensor) == Nx.iota({10, 10}, backend: Nx.BinaryBackend)
    end

    defn iota0, do: Nx.iota({10, 10}, axis: 0)

    test "iota0" do
      %{data: %Manx{}} = tensor = iota0()
      assert Nx.backend_transfer(tensor) == Nx.iota({10, 10}, axis: 0, backend: Nx.BinaryBackend)
    end

    defn iota1, do: Nx.iota({10, 10}, axis: 1)

    test "iota1" do
      %{data: %Manx{}} = tensor = iota1()
      assert Nx.backend_transfer(tensor) == Nx.iota({10, 10}, axis: 1, backend: Nx.BinaryBackend)
    end
  end

  describe "as_type" do
    defn to_float(t), do: Nx.as_type(t, {:f, 32})

    test "converts tensor type" do
      assert_equal(to_float(Nx.tensor([1, 2, 3])), Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}))
    end

    defn generic_as_type(t, template), do: Nx.as_type(t, template.type)

    test "converts non-finite types" do
      non_finite =
        Nx.stack([Nx.Constants.infinity(), Nx.Constants.nan(), Nx.Constants.neg_infinity()])

      # TODO: fix this, they should be equal to each other
      assert_not_equal(
        generic_as_type(non_finite, Nx.template({}, {:u, 8})),
        Nx.tensor([255, 0, 0], type: {:u, 8})
      )

      assert_not_equal(
        generic_as_type(non_finite, Nx.template({}, {:s, 16})),
        Nx.tensor([32767, 0, -32768], type: {:s, 16})
      )
    end
  end

  describe "bitcast" do
    defn bitcast_to_float(t), do: Nx.bitcast(t, {:f, 32})

    test "converts tensor type" do
      assert_equal(
        bitcast_to_float(Nx.tensor([0, 0, 0], type: {:s, 32})),
        Nx.tensor([0.0, 0.0, 0.0])
      )
    end
  end

  describe "squeeze" do
    defn squeeze(t), do: Nx.squeeze(t)
    defn squeeze2(t), do: Nx.squeeze(t, axes: [0, 1])

    test "with scalar" do
      assert_equal(squeeze(Nx.tensor(1)), Nx.tensor(1))
    end

    test "with tensors" do
      assert_equal(squeeze(Nx.tensor([[1, 2, 3]])), Nx.tensor([1, 2, 3]))
      assert_equal(squeeze(Nx.tensor([[[[[1]]]]])), Nx.tensor(1))
      assert_equal(squeeze2(Nx.tensor([[[[[1]]]]])), Nx.tensor([[[1]]]))
    end
  end

  @tag :runtime
  describe "slicing" do
    defn slice1(t), do: Nx.slice(t, [0, 6, 2], [2, 1, 3])

    defn slice1_dynamic(t), do: Nx.slice(t, [Nx.tensor(0), Nx.tensor(6), Nx.tensor(2)], [2, 1, 3])

    defn slice2(t), do: Nx.slice(t, [1, 4, 10], [1, 1, 10], strides: [1, 2, 3])

    defn slice2_dynamic(t),
      do: Nx.slice(t, [Nx.tensor(1), Nx.tensor(4), Nx.tensor(10)], [1, 1, 10], strides: [1, 2, 3])

    defn slice3(t), do: Nx.slice(t, [0, 4, 11], [2, 3, 9], strides: [2, 1, 3])

    defn slice3_dynamic(t),
      do: Nx.slice(t, [Nx.tensor(0), Nx.tensor(4), Nx.tensor(11)], [2, 3, 9], strides: [2, 1, 3])

    test "works without stride" do
      t = Nx.iota({900})
      t = Nx.reshape(t, {2, 15, 30})
      assert_equal(slice1(t), Nx.tensor([[[182, 183, 184]], [[632, 633, 634]]]))
      assert_equal(slice1_dynamic(t), Nx.tensor([[[182, 183, 184]], [[632, 633, 634]]]))
    end

    test "works with stride" do
      t = Nx.iota({900})
      t = Nx.reshape(t, {2, 15, 30})
      assert_equal(slice2(t), Nx.tensor([[[580, 583, 586, 589]]]))
      assert_equal(slice2_dynamic(t), Nx.tensor([[[580, 583, 586, 589]]]))

      assert_equal(
        slice3(t),
        Nx.tensor([
          [
            [131, 134, 137],
            [161, 164, 167],
            [191, 194, 197]
          ]
        ])
      )

      assert_equal(
        slice3_dynamic(t),
        Nx.tensor([
          [
            [131, 134, 137],
            [161, 164, 167],
            [191, 194, 197]
          ]
        ])
      )
    end
  end
end
