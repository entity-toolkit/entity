# cmake-lint: disable=C0103,C0111,E1120,R0913,R0915

cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0110 NEW)

set(PROJECT_NAME entity)

project(
  ${PROJECT_NAME}
  VERSION 1.4.2
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

set(deposit
    ${default_deposit}
    CACHE STRING "Deposit")

set(shape_order
    ${default_shape_order}
    CACHE STRING "Shape function")

set(pgen
    ${default_pgen}
    CACHE STRING "Problem generator")

set(output
    ${default_output}
    CACHE BOOL "Enable output")
set(mpi
    ${default_mpi}
    CACHE BOOL "Use MPI")

set(gpu_aware_mpi
    ${default_gpu_aware_mpi}
    CACHE BOOL "Enable GPU-aware MPI")

set(team_policy
  ${default_team_policy}
  CACHE BOOL "Enable team_policy tile-blocked deposit/pusher kernels")
set(team_policy_tile_size
  ${default_team_policy_tile_size}
  CACHE STRING "team_policy tile edge length in cells")
set(team_policy_tile_sizes
  "4;6;8;10;12"
  CACHE STRING "team_policy tile-size choices")

# -------------------------- Compilation settings -------------------------- #
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(${DEBUG} STREQUAL "OFF")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "CMake build type")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3")
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

set(deposits
    "zigzag" "esirkepov"
    CACHE STRING "Deposits")

if(${deposit} STREQUAL "zigzag")
  set(shape_order
      ${default_shape_order}
      CACHE STRING "Shape functions")
endif()

set(shape_orders
    "1;2;3;4;5;6;7;8;9;10;11"
    CACHE STRING "Shape orders")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)

# ------------------------- Third-Party Tests ------------------------------ #
set(BUILD_TESTING
    OFF
    CACHE BOOL "Build tests")

# ------------------------ Third-party dependencies ------------------------ #
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies.cmake)

find_or_fetch_dependency(Kokkos FALSE QUIET)
set(DEPENDENCIES Kokkos::kokkos)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# -------------------------------- Main code ------------------------------- #
set_precision(${precision})
set_shape_order(${shape_order})
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

# ------------------------------ team_policy wiring ------------------------ #
if(${team_policy})
  list(FIND team_policy_tile_sizes "${team_policy_tile_size}" _tps_idx)
  if(_tps_idx EQUAL -1)
    message(FATAL_ERROR
      "${Red}team_policy_tile_size must be one of ${team_policy_tile_sizes}, "
      "got '${team_policy_tile_size}'${ColorReset}")
  endif()
  add_compile_options("-D TEAM_POLICY")
  add_compile_options("-D TEAM_POLICY_TILE_SIZE=${team_policy_tile_size}")

  # Vendor sort: oneDPL on SYCL, Thrust on CUDA. Used automatically
  # when found; falls back to Kokkos::BinSort otherwise.
  if("${Kokkos_DEVICES}" MATCHES "SYCL")
    find_package(oneDPL QUIET)
    if(oneDPL_FOUND)
      message(STATUS "team_policy: oneDPL found, enabling SYCL sort_by_key")
      add_compile_options("-D ONEDPL_ENABLED")
      set(DEPENDENCIES ${DEPENDENCIES} oneDPL)
    else()
      message(STATUS "team_policy: oneDPL not found; using BinSort fallback "
                     "for SYCL sort_by_key")
    endif()
  endif()

  if("${Kokkos_DEVICES}" MATCHES "CUDA")
    find_package(Thrust QUIET)
    if(Thrust_FOUND)
      message(STATUS "team_policy: Thrust enabled for CUDA sort_by_key")
      add_compile_options("-D THRUST_ENABLED")
    else()
      message(STATUS "team_policy: Thrust not found; using BinSort fallback "
                     "for CUDA sort_by_key")
    endif()
  endif()

  if("${Kokkos_DEVICES}" MATCHES "HIP")
    # rocThrust ships with ROCm and exposes the same thrust:: API. Using
    # it lets the HIP backend build a single permutation via
    # sort_by_key and gather all SoA members through one reused buffer,
    # instead of the legacy per-member Kokkos::BinSort path which
    # allocates a fresh `sorted_values` buffer for every member every
    # step (the dominant source of allocator churn / fragmentation on
    # ROCm).
    find_package(rocthrust QUIET)
    if(rocthrust_FOUND)
      message(STATUS "team_policy: rocThrust enabled for HIP sort_by_key")
      add_compile_options("-D ROCTHRUST_ENABLED")
      set(DEPENDENCIES ${DEPENDENCIES} roc::rocthrust)
    else()
      message(STATUS "team_policy: rocThrust not found; using BinSort "
                     "fallback for HIP sort_by_key")
    endif()
  endif()
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
    set(DEPENDENCIES ${DEPENDENCIES} adios2::cxx_mpi)
  else()
    set(DEPENDENCIES ${DEPENDENCIES} adios2::cxx)
  endif()
endif()

link_libraries(${DEPENDENCIES})

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/)
add_subdirectory(${SRC_DIR}/global ${CMAKE_CURRENT_BINARY_DIR}/global)
add_subdirectory(${SRC_DIR}/metrics ${CMAKE_CURRENT_BINARY_DIR}/metrics)
add_subdirectory(${SRC_DIR}/kernels ${CMAKE_CURRENT_BINARY_DIR}/kernels)
add_subdirectory(${SRC_DIR}/archetypes ${CMAKE_CURRENT_BINARY_DIR}/archetypes)
add_subdirectory(${SRC_DIR}/framework ${CMAKE_CURRENT_BINARY_DIR}/framework)
add_subdirectory(${SRC_DIR}/output ${CMAKE_CURRENT_BINARY_DIR}/output)

# ------------------------------- Main source ------------------------------ #

if(NOT DEFINED pgens)
  set(single_pgen_mode ON)
  get_available_pgens(available_pgens)
  if(NOT ${pgen} STREQUAL ${default_pgen})
    set(pgen_suffix "")
    set_problem_generator(${pgen})
    add_subdirectory(${SRC_DIR}/engines ${CMAKE_CURRENT_BINARY_DIR}/engines)
    add_subdirectory(${SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/src)
  endif()
  list(APPEND problem_generators ${available_pgens})
else()
  set(single_pgen_mode OFF)
  get_available_pgens(available_pgens)
  set(pgens_short "")
  foreach(pg ${pgens})
    string(REPLACE "./" "" pg_nodir ${pg})
    string(REPLACE "/" "_" pg_nodir ${pg_nodir})
    set(pgen_suffix "_${pg_nodir}")
    set_problem_generator(${pg})
    add_subdirectory(${SRC_DIR}/engines
                     ${CMAKE_CURRENT_BINARY_DIR}/${pg_nodir}/engines)
    add_subdirectory(${SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${pg_nodir}/src)
    list(APPEND pgens_short ${PGEN})
  endforeach()
  list(APPEND problem_generators ${available_pgens})
endif()

if(TESTS)
  # ---------------------------------- Tests --------------------------------- #
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests.cmake)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/report.cmake)
