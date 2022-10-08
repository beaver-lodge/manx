# 🐱 Manx 🐈

**M**LIR-**A**ccelerated-**Nx**. MLIR compiler/backend for the [Nx](https://github.com/elixir-nx/nx/tree/main/nx#readme).

## Why do we need it?

Instead of repurposing compilers built for Python, Manx is about building a Nx compiler in Elixir and tailored for Elixir.
"Tensor compiler" is no longer a giant black box for us anymore. We can build passes and optimizations for Elixir and BEAM and even generate LLVM instructions to allocate memory with Erlang's allocator.

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `manx` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:manx, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/manx>.
