# ------------------------------
# @defines: ntt_archetypes [INTERFACE]
# @includes:
# - ../
# @depends:
# - ntt_global [required]
# @uses:
# - kokkos [required]
# - mpi [optional]
# ------------------------------

add_library(ntt_archetypes INTERFACE)
target_link_libraries(ntt_archetypes INTERFACE ntt_global)
target_include_directories(ntt_archetypes
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../
)