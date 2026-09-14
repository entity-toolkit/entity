# cmake-lint: disable=C0103
# ------------------------------
# @defines: ntt_framework [STATIC/SHARED]
#
# @sources:
#
# * parameters/parameters.cpp
# * parameters/particles.cpp
# * parameters/grid.cpp
# * parameters/output.cpp
# * parameters/algorithms.cpp
# * parameters/extra.cpp
# * simulation.cpp
# * domain/grid.cpp
# * domain/metadomain.cpp
# * domain/metadomain_sort.cpp
# * domain/metadomain_stats.cpp
# * domain/metadomain_reshape.cpp
# * domain/metadomain_loadbal.cpp
# * domain/checkpoint/init.cpp
# * domain/checkpoint/write.cpp
# * domain/checkpoint/resume.cpp
# * domain/comm/fields.cpp
# * domain/comm/fields_sync.cpp
# * domain/comm/particles.cpp
# * domain/comm/vector_potential.cpp
# * domain/io/init.cpp
# * domain/io/write.cpp
# * domain/io/spectra.cpp
# * domain/io/fields.cpp
# * containers/particles.cpp
# * containers/particles_sort.cpp
# * containers/fields.cpp
# * containers/comm/particles.cpp
# * containers/io/particles.cpp
# * containers/checkpoint/particles.cpp
# * containers/checkpoint/fields.cpp
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
    ${SRC_DIR}/simulation.cpp
    ${SRC_DIR}/parameters/parameters.cpp
    ${SRC_DIR}/parameters/particles.cpp
    ${SRC_DIR}/parameters/grid.cpp
    ${SRC_DIR}/parameters/output.cpp
    ${SRC_DIR}/parameters/algorithms.cpp
    ${SRC_DIR}/parameters/extra.cpp
    ${SRC_DIR}/domain/grid.cpp
    ${SRC_DIR}/domain/metadomain.cpp
    ${SRC_DIR}/domain/metadomain_sort.cpp
    ${SRC_DIR}/domain/metadomain_stats.cpp
    ${SRC_DIR}/domain/metadomain_reshape.cpp
    ${SRC_DIR}/domain/metadomain_loadbal.cpp
    ${SRC_DIR}/domain/comm/fields.cpp
    ${SRC_DIR}/domain/comm/fields_sync.cpp
    ${SRC_DIR}/domain/comm/particles.cpp
    ${SRC_DIR}/containers/particles.cpp
    ${SRC_DIR}/containers/particles_sort.cpp
    ${SRC_DIR}/containers/fields.cpp)
if(${output})
  list(
    APPEND
    SOURCES
    ${SRC_DIR}/domain/io/init.cpp
    ${SRC_DIR}/domain/io/write.cpp
    ${SRC_DIR}/domain/io/spectra.cpp
    ${SRC_DIR}/domain/io/fields.cpp
    ${SRC_DIR}/domain/checkpoint/init.cpp
    ${SRC_DIR}/domain/checkpoint/write.cpp
    ${SRC_DIR}/domain/checkpoint/resume.cpp
    ${SRC_DIR}/containers/checkpoint/fields.cpp
    ${SRC_DIR}/containers/checkpoint/particles.cpp
    ${SRC_DIR}/containers/io/particles.cpp
    ${SRC_DIR}/domain/comm/vector_potential.cpp)
endif()
if(${mpi})
  list(APPEND SOURCES ${SRC_DIR}/containers/comm/particles.cpp)
endif()
add_library(ntt_framework ${SOURCES})

set(libs ntt_global ntt_metrics ntt_kernels ntt_output)
add_dependencies(ntt_framework ${libs})
target_link_libraries(ntt_framework PUBLIC ${libs})
if(NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "AppleClang")
  target_link_libraries(ntt_framework PRIVATE stdc++fs)
endif()

target_include_directories(
  ntt_framework
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)
