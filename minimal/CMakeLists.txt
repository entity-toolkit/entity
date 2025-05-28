# cmake-lint: disable=C0103,C0111,E1120,R0913,R0915
cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0110 NEW)

set(PROJECT_NAME minimal-test)

project(${PROJECT_NAME} LANGUAGES CXX C)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if($DEBUG)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "CMake build type")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
else()
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "CMake build type")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -DDEBUG -Wall -Wextra -Wno-unknown-pragmas")
endif()

set(BUILD_TESTING
    OFF
    CACHE BOOL "Build tests")

set(MODES
    "KOKKOS;ADIOS2_NOMPI"
    CACHE STRING "Build modes")

function(find_kokkos)
  find_package(Kokkos QUIET)
  if(NOT Kokkos_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      Kokkos
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG 4.6.01)
    FetchContent_MakeAvailable(Kokkos)
  endif()
  if(NOT DEFINED Kokkos_ARCH
     OR Kokkos_ARCH STREQUAL ""
     OR NOT DEFINED Kokkos_DEVICES
     OR Kokkos_DEVICES STREQUAL "")
    if(${Kokkos_FOUND})
      include(${Kokkos_DIR}/KokkosConfigCommon.cmake)
    elseif(NOT ${Kokkos_BUILD_DIR} STREQUAL "")
      include(${Kokkos_BUILD_DIR}/KokkosConfigCommon.cmake)
    else()
      message(
        STATUS "${Red}Kokkos_DIR and Kokkos_BUILD_DIR not set.${ColorReset}")
    endif()
  endif()
endfunction()

function(find_adios2)
  find_package(adios2 QUIET)
  if(NOT adios2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      adios2
      GIT_REPOSITORY https://github.com/ornladios/ADIOS2.git
      GIT_TAG 2.10.2)
    FetchContent_MakeAvailable(adios2)
  endif()
endfunction()

if("KOKKOS" IN_LIST MODES)
  set(libs "")
  set(exec kokkos.xc)
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/kokkos.cpp)

  find_kokkos()
  list(APPEND libs Kokkos::kokkos)

  add_executable(${exec} ${src})

  target_link_libraries(${exec} ${libs})
endif()

if("ADIOS2_NOMPI" IN_LIST MODES)
  set(libs stdc++fs)
  set(exec adios2-nompi.xc)
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/adios2.cpp)

  find_kokkos()
  find_adios2()
  list(APPEND libs Kokkos::kokkos adios2::cxx11)

  add_executable(${exec} ${src})

  target_link_libraries(${exec} ${libs})
endif()

if("ADIOS2_MPI" IN_LIST MODES)
  set(libs stdc++fs)
  set(exec adios2-mpi.xc)
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/adios2.cpp)

  find_package(MPI REQUIRED)
  find_kokkos()
  find_adios2()
  list(APPEND libs MPI::MPI_CXX Kokkos::kokkos adios2::cxx11_mpi)

  add_executable(${exec} ${src})

  target_include_directories(${exec} PUBLIC ${MPI_CXX_INCLUDE_PATH})
  target_compile_options(${exec} PUBLIC "-D MPI_ENABLED")
  target_link_libraries(${exec} ${libs})
endif()

if("MPI" IN_LIST MODES)
  set(libs "")
  set(exec mpi-simple.xc)
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/mpi-simple.cpp)

  find_package(MPI REQUIRED)
  find_kokkos()
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
    target_compile_options(${exec} PRIVATE -DDEVICE_ENABLED)
  else()
    set(DEVICE_ENABLED OFF)
  endif()

  if(${GPU_AWARE_MPI})
    target_compile_options(${exec} PRIVATE -DGPU_AWARE_MPI)
  endif()
endif()

if("MPI_SIMPLE" IN_LIST MODES)
  set(libs "")
  set(exec mpi-simple.xc)
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/mpi-simple.cpp)

  find_package(MPI REQUIRED)
  find_kokkos()
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
    target_compile_options(${exec} PRIVATE -DDEVICE_ENABLED)
  else()
    set(DEVICE_ENABLED OFF)
  endif()

  if(${GPU_AWARE_MPI})
    target_compile_options(${exec} PRIVATE -DGPU_AWARE_MPI)
  endif()
endif()

message(STATUS "Build modes: ${MODES}")
