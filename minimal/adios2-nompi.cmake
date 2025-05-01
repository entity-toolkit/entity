# cmake-lint: disable=C0103
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/dependencies.cmake)

set(libs stdc++fs)
set(exec adios2-nompi.xc)
set(src ${CMAKE_CURRENT_SOURCE_DIR}/adios2.cpp)

find_or_fetch_dependency(Kokkos FALSE QUIET)
find_or_fetch_dependency(adios2 FALSE QUIET)
list(APPEND libs Kokkos::kokkos adios2::cxx11)

add_executable(${exec} ${src})

target_link_libraries(${exec} ${libs})
