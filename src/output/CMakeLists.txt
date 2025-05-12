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

set(SOURCES ${SRC_DIR}/stats.cpp ${SRC_DIR}/fields.cpp
            ${SRC_DIR}/utils/interpret_prompt.cpp)
if(${output})
  list(APPEND SOURCES ${SRC_DIR}/writer.cpp)
endif()
add_library(ntt_output ${SOURCES})

set(libs ntt_global)
add_dependencies(ntt_output ${libs})
target_link_libraries(ntt_output PUBLIC ${libs})
target_link_libraries(ntt_output PRIVATE stdc++fs)

target_include_directories(
  ntt_output
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)
