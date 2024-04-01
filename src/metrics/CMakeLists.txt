# ------------------------------
# @defines: ntt_metrics [INTERFACE]
# @includes:
# - ./
# @depends:
# - ntt_global [required]
# @uses:
# - kokkos [required]
# ------------------------------

add_library(ntt_metrics INTERFACE)
target_link_libraries(ntt_metrics INTERFACE ntt_global)
target_include_directories(ntt_metrics
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../
)