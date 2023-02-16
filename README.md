# 🐱 Manx 🐈

**M**LIR-**A**ccelerated-**Nx**. MLIR compiler/backend for the [Nx](https://github.com/elixir-nx/nx/tree/main/nx#readme).

## What does this library do?

You can think of Manx as IREE implemented in Elixir and unlike IREE with new dedicated runtime Manx uses BEAM as the runtime. Nx's expressions are very close to XLA's MHLO so Manx would borrow a lot of conversion/lowering implementations from [XLA](https://github.com/openxla/xla) and [IREE](https://github.com/iree-org/iree).

## Why do we need it?

- Instead of repurposing compilers built for Python, Manx is about building a Nx compiler in Elixir and tailored for Elixir.
- With Manx, "Tensor compiler" is no longer a giant black box for Erlang world anymore. A non-python programming language should have its full-stack data/ML solution so that it could be truly maintainable.
- Tighter integration with BEAM. We can build passes and optimizations for Elixir and BEAM and even generate LLVM instructions to send messages or allocate memory with Erlang's allocator.
- There is a great gap between the understanding "distributed system" in ML and non-ML applications (MPI vs. fault-tolerance). With Manx we could narrow the gap by implementing a ML compiler with a programming language with strong fault-tolerance capability.

## Compared to EXLA

- [EXLA](https://github.com/elixir-nx/nx/tree/main/exla) is the Nx backend for XLA.
- In the short run, Manx's performance won't be on-per with XLA/EXLA's.

- EXLA's lowering:

```
Nx |> EXLA |> XLA |> MLIR |> LLVM |> hardware
```

- Manx's lowering

```
Nx |> Manx |> MLIR |> LLVM |> hardware
```

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
