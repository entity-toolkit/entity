# ------------------------------
# @defines: ntt_kernels [INTERFACE]
# @includes:
# - ../
# @depends:
# - ntt_global [required]
# @uses:
# - kokkos [required]
# - plog [required]
# - mpi [optional]
# ------------------------------

add_library(ntt_kernels INTERFACE)
target_link_libraries(ntt_kernels INTERFACE ntt_global)
target_include_directories(ntt_kernels
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../
)