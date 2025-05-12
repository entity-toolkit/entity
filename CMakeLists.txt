# cmake-lint: disable=C0103,C0111,E1120,R0913,R0915

cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0110 NEW)

set(PROJECT_NAME entity)

project(
  ${PROJECT_NAME}
  VERSION 1.2.0
  LANGUAGES CXX C)
add_compile_options("-D ENTITY_VERSION=\"${PROJECT_VERSION}\"")
set(hash_cmd "git diff --quiet src/ && echo $(git rev-parse HEAD) ")
string(APPEND hash_cmd "|| echo $(git rev-parse HEAD)-mod")
execute_process(
  COMMAND bash -c ${hash_cmd}
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_HASH
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Git hash: ${GIT_HASH}")
add_compile_options("-D ENTITY_GIT_HASH=\"${GIT_HASH}\"")

set(CMAKE_CXX_EXTENSIONS OFF)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/styling.cmake)

# ----------------------------- Configurations ----------------------------- #
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/defaults.cmake)

# defaults
set(DEBUG
    ${default_debug}
    CACHE BOOL "Debug mode")

set(precision
    ${default_precision}
    CACHE STRING "Precision")
set(pgen
    ${default_pgen}
    CACHE STRING "Problem generator")

set(gui
    ${default_gui}
    CACHE BOOL "Use GUI [nttiny]")
set(output
    ${default_output}
    CACHE BOOL "Enable output")
set(mpi
    ${default_mpi}
    CACHE BOOL "Use MPI")

set(gpu_aware_mpi
    ${default_gpu_aware_mpi}
    CACHE BOOL "Enable GPU-aware MPI")

# -------------------------- Compilation settings -------------------------- #
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(${DEBUG} STREQUAL "OFF")
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

# options
set(precisions
    "single" "double"
    CACHE STRING "Precisions")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)

# ------------------------- Third-Party Tests ------------------------------ #
set(BUILD_TESTING
    OFF
    CACHE BOOL "Build tests")

# ------------------------ Third-party dependencies ------------------------ #
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies.cmake)

find_or_fetch_dependency(Kokkos FALSE QUIET)
find_or_fetch_dependency(plog TRUE QUIET)
set(DEPENDENCIES Kokkos::kokkos)
include_directories(${plog_SRC}/include)

# -------------------------------- Main code ------------------------------- #
set_precision(${precision})
if("${Kokkos_DEVICES}" MATCHES "CUDA")
  add_compile_options("-D CUDA_ENABLED")
  set(DEVICE_ENABLED ON)
  add_compile_options("-D DEVICE_ENABLED")
elseif("${Kokkos_DEVICES}" MATCHES "HIP")
  add_compile_options("-D HIP_ENABLED")
  set(DEVICE_ENABLED ON)
  add_compile_options("-D DEVICE_ENABLED")
elseif("${Kokkos_DEVICES}" MATCHES "SYCL")
  add_compile_options("-D SYCL_ENABLED")
  set(DEVICE_ENABLED ON)
  add_compile_options("-D DEVICE_ENABLED")
else()
  set(DEVICE_ENABLED OFF)
endif()

if(("${Kokkos_DEVICES}" MATCHES "CUDA")
   OR ("${Kokkos_DEVICES}" MATCHES "HIP")
   OR ("${Kokkos_DEVICES}" MATCHES "SYCL"))
  set(DEVICE_ENABLED ON)
else()
  set(DEVICE_ENABLED OFF)
endif()

# MPI
if(${mpi})
  find_or_fetch_dependency(MPI FALSE REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  add_compile_options("-D MPI_ENABLED")
  set(DEPENDENCIES ${DEPENDENCIES} MPI::MPI_CXX)
  if(${DEVICE_ENABLED})
    if(${gpu_aware_mpi})
      add_compile_options("-D GPU_AWARE_MPI")
    endif()
  else()
    set(gpu_aware_mpi
        OFF
        CACHE BOOL "Use explicit copy when using MPI + GPU")
  endif()
endif()

# Output
if(${output})
  find_or_fetch_dependency(adios2 FALSE QUIET)
  add_compile_options("-D OUTPUT_ENABLED")
  if(${mpi})
    set(DEPENDENCIES ${DEPENDENCIES} adios2::cxx11_mpi)
  else()
    set(DEPENDENCIES ${DEPENDENCIES} adios2::cxx11)
  endif()
endif()

link_libraries(${DEPENDENCIES})

if(TESTS)
  # ---------------------------------- Tests --------------------------------- #
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests.cmake)
elseif(BENCHMARK)
  # ------------------------------ Benchmark --------------------------------- #
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/benchmark.cmake)
else()
  # ----------------------------------- GUI ---------------------------------- #
  if(${gui})
    find_or_fetch_dependency(nttiny FALSE QUIET)
  endif()

  # ------------------------------- Main source ------------------------------ #
  set_problem_generator(${pgen})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src src)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/report.cmake)
