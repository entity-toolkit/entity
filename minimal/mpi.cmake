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

set(GPU_AWARE_MPI
    ON
    CACHE BOOL "Enable GPU-aware MPI support")

if(("${Kokkos_DEVICES}" MATCHES "CUDA")
   OR ("${Kokkos_DEVICES}" MATCHES "HIP")
   OR ("${Kokkos_DEVICES}" MATCHES "SYCL"))
  set(DEVICE_ENABLED ON)
  target_copmile_options(${exec} PRIVATE -DDEVICE_ENABLED)
else()
  set(DEVICE_ENABLED OFF)
endif()

if(${GPU_AWARE_MPI})
  target_compile_options(${exec} PRIVATE -DGPU_AWARE_MPI)
endif()
