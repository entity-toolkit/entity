# cmake-lint: disable=C0103
# ------------------------------
# @defines: ntt_output [STATIC/SHARED]
#
# @sources:
#
# * writer.cpp
# * fields.cpp
# * stats.cpp
# * utils/interpret_prompt.cpp
# * utils/writers.cpp
# * utils/readers.cpp
# * utils/tuning.cpp
# * render/renderer.cpp
#
# @includes:
#
# * ../
#
# @depends:
#
# * ntt_global [required]
#
# @uses:
#
# * kokkos [required]
# * ADIOS2 [optional]
# * mpi [optional]
# ------------------------------

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCES
    ${SRC_DIR}/stats.cpp ${SRC_DIR}/fields.cpp
    ${SRC_DIR}/utils/interpret_prompt.cpp ${SRC_DIR}/render/renderer.cpp)
if(${output})
  list(
    APPEND
    SOURCES
    ${SRC_DIR}/writer.cpp
    ${SRC_DIR}/checkpoint.cpp
    ${SRC_DIR}/utils/writers.cpp
    ${SRC_DIR}/utils/readers.cpp
    ${SRC_DIR}/utils/tuning.cpp)
endif()
add_library(ntt_output ${SOURCES})

set(libs ntt_global)
add_dependencies(ntt_output ${libs})
target_link_libraries(ntt_output PUBLIC ${libs})
if(NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "AppleClang")
  target_link_libraries(ntt_output PRIVATE stdc++fs)
endif()

target_include_directories(
  ntt_output
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)
