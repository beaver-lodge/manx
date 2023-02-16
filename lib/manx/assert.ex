defmodule Manx.Assert do
  import Nx.Defn

  @moduledoc """
  Tensor assertions. Original implementation is from EXLA.
  """

  defmacro assert_equal(left, right) do
    # Assert against binary backend tensors to show diff on failure
    quote do
      assert unquote(left) |> Manx.Assert.to_binary_backend() ==
               unquote(right) |> Manx.Assert.to_binary_backend()
    end
  end

  defmacro assert_not_equal(left, right) do
    # Assert against binary backend tensors to show diff on failure
    quote do
      assert unquote(left) |> Manx.Assert.to_binary_backend() !=
               unquote(right) |> Manx.Assert.to_binary_backend()
    end
  end

  def to_binary_backend(tensor) do
    Nx.backend_copy(tensor, Nx.BinaryBackend)
  end

  defn all_close_jit(a, b, opts \\ []) do
    import Nx
    opts = keyword!(opts, equal_nan: false, rtol: 1.0e-5, atol: 1.0e-8, both_integer: false)
    both_integer = opts[:both_integer]
    rtol = opts[:rtol]
    atol = opts[:atol]

    a = to_tensor(a)
    b = to_tensor(b)

    finite_entries = less_equal(Nx.abs(subtract(a, b)), add(atol, multiply(rtol, Nx.abs(b))))

    if both_integer do
      all(finite_entries)
    else
      # inf - inf is a nan, however, they are equal,
      # so we explicitly check for equal entries.
      inf_a = is_infinity(a)
      inf_b = is_infinity(b)
      inf_entries = select(logical_or(inf_a, inf_b), equal(a, b), finite_entries)

      if opts[:equal_nan] do
        nan_a = is_nan(a)
        nan_b = is_nan(b)
        nan_entries = logical_and(nan_a, nan_b)
        all(select(nan_entries, 1, inf_entries))
      else
        all(inf_entries)
      end
    end
  end

  # def all_close(left, right, opts \\ []) do
  #   true
  # end

  def all_close(left, right, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-4)
    rtol = Keyword.get(opts, :rtol, 1.0e-4)

    equals =
      left
      |> all_close_jit(right,
        atol: atol,
        rtol: rtol,
        both_integer: Nx.Type.integer?(left.type) and Nx.Type.integer?(right.type)
      )
      |> Nx.backend_transfer(Nx.BinaryBackend)

    if equals != Nx.tensor(1, type: {:u, 8}, backend: Nx.BinaryBackend) do
      raise("""
      expected

      #{inspect(left)}

      to be within tolerance of

      #{inspect(right)}
      """)
    end
  end

  defmacro assert_all_close(left, right, opts \\ []) do
    quote bind_quoted: [
            left: left,
            right: right,
            opts: opts
          ] do
      all_close(left, right, opts)
    end
  end

  def evaluate(fun, args) do
    fun |> Nx.Defn.jit(compiler: Nx.Defn.Evaluator) |> apply(args)
  end
end
