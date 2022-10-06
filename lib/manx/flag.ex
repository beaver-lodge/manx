defmodule Manx.Flags do
  def print_ir?() do
    System.get_env("MANX_PRINT_IR") == "1"
  end
end
