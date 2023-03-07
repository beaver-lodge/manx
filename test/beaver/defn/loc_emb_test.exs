defmodule TestEmbedding do
  @moduledoc """
  embedding implementation from bumblebee
  """
  import Nx.Defn

  defn timestep_sinusoidal_embedding_impl(timestep, opts \\ []) do
    opts =
      keyword!(opts, [
        :embedding_size,
        flip_sin_to_cos: false,
        frequency_correction_term: 1,
        scale: 1,
        max_period: 10_000,
        mode: :train
      ])

    embedding_size = opts[:embedding_size]
    max_period = opts[:max_period]
    frequency_correction_term = opts[:frequency_correction_term]

    if rem(embedding_size, 2) != 0 do
      raise ArgumentError,
            "expected embedding size to an even number, but got: #{inspect(embedding_size)}"
    end

    half_size = div(embedding_size, 2)

    frequency =
      Nx.exp(-Nx.log(max_period) * Nx.iota({half_size}) / (half_size - frequency_correction_term))

    angle = Nx.new_axis(timestep, -1) * Nx.new_axis(frequency, 0)
    angle = opts[:scale] * angle

    if opts[:flip_sin_to_cos] do
      Nx.concatenate([Nx.cos(angle), Nx.sin(angle)], axis: -1)
    else
      Nx.concatenate([Nx.sin(angle), Nx.cos(angle)], axis: -1)
    end
  end
end

defmodule Beaver.Defn.LocEmbTest do
  use ExUnit.Case, async: true

  @moduletag :nx
  @moduletag :runtime
  setup do
    Nx.Defn.default_options(compiler: Manx.Compiler)
    :ok
  end

  test "time emb" do
    TestEmbedding.timestep_sinusoidal_embedding_impl(100, embedding_size: 10)
  end
end
