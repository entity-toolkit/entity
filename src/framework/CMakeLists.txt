# ------------------------------
# @defines: ntt_framework [STATIC/SHARED]
# @sources:
# - parameters.cpp
# @includes:
# - ../
# @depends:
# - ntt_global [required]
# - ntt_metrics [required]
# @uses:
# - kokkos [required]
# - plog [required]
# - toml11 [required]
# - ADIOS2 [optional]
# - mpi [optional]
# ------------------------------

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SOURCES ${SRC_DIR}/parameters.cpp)
add_library(ntt_framework ${SOURCES})
target_link_libraries(ntt_framework PUBLIC ntt_global ntt_metrics)
target_include_directories(ntt_framework
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../
)