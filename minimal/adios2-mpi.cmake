# cmake-lint: disable=C0103
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/dependencies.cmake)

set(libs stdc++fs)
set(exec adios2-mpi.xc)
set(src ${CMAKE_CURRENT_SOURCE_DIR}/adios2.cpp)

find_or_fetch_dependency(MPI FALSE REQUIRED)
find_or_fetch_dependency(Kokkos FALSE QUIET)
find_or_fetch_dependency(adios2 FALSE QUIET)
list(APPEND libs MPI::MPI_CXX Kokkos::kokkos adios2::cxx11_mpi)

add_executable(${exec} ${src})

target_include_directories(${exec} PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_compile_options(${exec} PUBLIC "-D MPI_ENABLED")
target_link_libraries(${exec} ${libs})
