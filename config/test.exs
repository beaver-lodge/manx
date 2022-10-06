import Config

config :beaver,
  skip_dialects: ~w{nvgpu
  x86vector
  vector
  omp
  emitc
  sparse_tensor
  amdgpu
  async
  llvm
  transform
  ml_program
  amx
  arm_neon
  spv
  quant
  arm_sve
  rocdl
  acc
  shape
  nvvm}
