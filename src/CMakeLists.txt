# cmake-lint: disable=C0103
# ------------------------------
# @defines: entity [STATIC/SHARED]
#
# @sources:
#
# * entity.cpp
#
# @depends:
#
# * ntt_global [required]
# * ntt_framework [required]
# * ntt_metrics [required]
# * ntt_engine [required]
# * ntt_pgen [required]
#
# @uses:
#
# * kokkos [required]
# * plog [required]
# * ADIOS2 [optional]
# * mpi [optional]
# ------------------------------

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# dependencies
add_subdirectory(${SRC_DIR}/global ${CMAKE_CURRENT_BINARY_DIR}/global)
add_subdirectory(${SRC_DIR}/metrics ${CMAKE_CURRENT_BINARY_DIR}/metrics)
add_subdirectory(${SRC_DIR}/kernels ${CMAKE_CURRENT_BINARY_DIR}/kernels)
add_subdirectory(${SRC_DIR}/archetypes ${CMAKE_CURRENT_BINARY_DIR}/archetypes)
add_subdirectory(${SRC_DIR}/framework ${CMAKE_CURRENT_BINARY_DIR}/framework)
add_subdirectory(${SRC_DIR}/engines ${CMAKE_CURRENT_BINARY_DIR}/engines)
add_subdirectory(${SRC_DIR}/output ${CMAKE_CURRENT_BINARY_DIR}/output)

if(${output})
  add_subdirectory(${SRC_DIR}/checkpoint ${CMAKE_CURRENT_BINARY_DIR}/checkpoint)
endif()

set(ENTITY ${PROJECT_NAME}.xc)
set(SOURCES ${SRC_DIR}/entity.cpp)

add_executable(${ENTITY} ${SOURCES})
set(libs ntt_global ntt_framework ntt_metrics ntt_engines ntt_pgen)
add_dependencies(${ENTITY} ${libs})
target_link_libraries(${ENTITY} PUBLIC ${libs})
