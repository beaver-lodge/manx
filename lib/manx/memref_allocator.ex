defmodule Manx.MemrefAllocator do
  @moduledoc """
  MemrefAllocator is an Agent managing memrefs.
  """
  use Agent

  def start_link(_) do
    Agent.start_link(
      fn -> :ets.new(__MODULE__, [:named_table, :public, read_concurrency: true]) end,
      name: __MODULE__
    )
  end

  def add(memref) do
    :ets.insert(__MODULE__, {memref})
  end

  def delete(memref) do
    found = :ets.lookup(__MODULE__, memref)

    if length(found) >= 1 do
      :ets.delete(__MODULE__, memref)
      :ok
    else
      :already_deallocated
    end
  end
end
