# ----------------------------- Adios2 settings ---------------------------- #
set(ADIOS2_BUILD_EXAMPLES OFF CACHE BOOL "Build ADIOS2 examples")

# Language support
set(ADIOS2_USE_Python OFF CACHE BOOL "Use Python for ADIOS2")
set(ADIOS2_USE_Fortran OFF CACHE BOOL "Use Fortran for ADIOS2")

# Format/compression support
set(ADIOS2_USE_ZeroMQ OFF CACHE BOOL "Use ZeroMQ for ADIOS2")
set(ADIOS2_USE_SST OFF CACHE BOOL "Use SST for ADIOS2")
set(ADIOS2_USE_BZip2 OFF CACHE BOOL "Use BZip2 for ADIOS2")
set(ADIOS2_USE_ZFP OFF CACHE BOOL "Use ZFP for ADIOS2")
set(ADIOS2_USE_SZ OFF CACHE BOOL "Use SZ for ADIOS2")
set(ADIOS2_USE_MGARD OFF CACHE BOOL "Use MGARD for ADIOS2")
set(ADIOS2_USE_PNG OFF CACHE BOOL "Use PNG for ADIOS2")
set(ADIOS2_USE_Blosc OFF CACHE BOOL "Use Blosc for ADIOS2")

# !TODO: add MPI-enabled ADIOS2
set(ADIOS2_USE_MPI OFF CACHE BOOL "Use MPI for ADIOS2")

# !TODO: add CUDA-enabled ADIOS2
set(ADIOS2_USE_CUDA OFF CACHE BOOL "Use CUDA for ADIOS2")

# set(ADIOS2_USE_CUDA ${Kokkos_ENABLE_CUDA} CACHE BOOL "Use CUDA for ADIOS2")
add_compile_options("-D OUTPUT_ENABLED")