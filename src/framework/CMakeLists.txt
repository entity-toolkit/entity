# cmake-lint: disable=C0103
# ------------------------------
# @defines: ntt_framework [STATIC/SHARED]
#
# @sources:
#
# * parameters.cpp
# * simulation.cpp
# * domain/grid.cpp
# * domain/metadomain.cpp
# * domain/communications.cpp
# * domain/checkpoint.cpp
# * containers/particles.cpp
# * containers/fields.cpp
# * domain/stats.cpp
# * domain/output.cpp
#
# @includes:
#
# * ../
#
# @depends:
#
# * ntt_global [required]
# * ntt_metrics [required]
# * ntt_kernels [required]
# * ntt_output [required]
#
# @uses:
#
# * kokkos [required]
# * plog [required]
# * ADIOS2 [optional]
# * mpi [optional]
# ------------------------------

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SOURCES
    ${SRC_DIR}/parameters.cpp
    ${SRC_DIR}/simulation.cpp
    ${SRC_DIR}/domain/grid.cpp
    ${SRC_DIR}/domain/metadomain.cpp
    ${SRC_DIR}/domain/communications.cpp
    ${SRC_DIR}/domain/stats.cpp
    ${SRC_DIR}/containers/particles.cpp
    ${SRC_DIR}/containers/fields.cpp)
if(${output})
  list(APPEND SOURCES ${SRC_DIR}/domain/output.cpp)
  list(APPEND SOURCES ${SRC_DIR}/domain/checkpoint.cpp)
endif()
add_library(ntt_framework ${SOURCES})

set(libs ntt_global ntt_metrics ntt_kernels ntt_output)
if(${output})
  list(APPEND libs ntt_checkpoint)
endif()
add_dependencies(ntt_framework ${libs})
target_link_libraries(ntt_framework PUBLIC ${libs})
target_link_libraries(ntt_framework PRIVATE stdc++fs)

target_include_directories(
  ntt_framework
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)
