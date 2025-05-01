# cmake-lint: disable=C0103
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/dependencies.cmake)

set(exec mpi.xc)
set(src ${CMAKE_CURRENT_SOURCE_DIR}/mpi.cpp)

find_or_fetch_dependency(MPI FALSE REQUIRED)
find_or_fetch_dependency(Kokkos FALSE QUIET)
list(APPEND libs MPI::MPI_CXX Kokkos::kokkos)

add_executable(${exec} ${src})

target_include_directories(${exec} PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(${exec} ${libs})
