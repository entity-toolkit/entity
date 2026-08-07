# cmake-lint: disable=C0103,C0111,E1120,R0913,R0915

cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0110 NEW)

set(PROJECT_NAME entity)

project(
  ${PROJECT_NAME}
  VERSION 1.5.0
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
set(ascent
    ${default_ascent}
    CACHE BOOL "Enable Ascent in situ visualization")
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
  "4;6;8;10;12;14;16"
  CACHE STRING "team_policy tile-size choices")
set(team_policy_drift
  ${default_team_policy_drift}
  CACHE STRING
  "team_policy tiled-deposit scratch halo drift in cells (max cells a particle may move between two sorts). Sizes the deposit scratch halo only; the sort cadence is set at runtime via spatial_sorting_interval. Default 1.")
set(vendor_sort
  ${default_vendor_sort}
  CACHE BOOL
  "Use the vendor sort_by_key (oneDPL/Thrust/rocThrust) for the team_policy spatial sort when available. OFF forces the Kokkos::BinSort fallback, which sorts each SoA member in place (lower peak memory, no maxnpart gather buffer) at the cost of sort speed.")

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

  # Compile-time tiled-deposit scratch halo drift. Sizes the halo so a
  # particle that drifts up to DRIFT cells between two sorts still deposits
  # inside its tile scratch; particles drifting further take the
  # per-particle global-J escape valve (correct, only slower). This is
  # independent of the sort cadence, which is set at runtime via
  # `spatial_sorting_interval`. Defaults to 1 (the sorted-every-step case).
  add_compile_options("-D TEAM_POLICY_DRIFT=${team_policy_drift}")

  # Vendor sort: oneDPL on SYCL, Thrust on CUDA, rocThrust/rocprim on HIP.
  # When `vendor_sort` is ON (default) the available library is detected
  # and used; the spatial sort then builds a single permutation that
  # gathers all SoA members. When `vendor_sort` is OFF, or no library is
  # found, the code falls back to Kokkos::BinSort, which sorts each member
  # in place -- lower peak memory and no maxnpart gather buffer, at the
  # cost of sort speed (negligible when sorting is a small fraction of the
  # step). The `vendor_sort` knob lets you force the BinSort fallback even
  # when a vendor library is present.
  if(${vendor_sort})
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
      # rocThrust ships with ROCm. The HIP sort_by_key path uses rocprim's
      # bounded-bit radix sort directly (rocprim is rocThrust's own
      # dependency, so its headers come in transitively; we find it
      # explicitly to keep the include path robust). This builds a single
      # permutation that gathers all SoA members, instead of the legacy
      # per-member Kokkos::BinSort path which allocates a fresh
      # `sorted_values` buffer for every member every step (the dominant
      # source of allocator churn / fragmentation on ROCm).
      find_package(rocthrust QUIET)
      if(rocthrust_FOUND)
        message(STATUS "team_policy: rocThrust enabled for HIP sort_by_key")
        add_compile_options("-D ROCTHRUST_ENABLED")
        set(DEPENDENCIES ${DEPENDENCIES} roc::rocthrust)
        find_package(rocprim QUIET)
        if(rocprim_FOUND)
          set(DEPENDENCIES ${DEPENDENCIES} roc::rocprim)
        endif()
      else()
        message(STATUS "team_policy: rocThrust not found; using BinSort "
                       "fallback for HIP sort_by_key")
      endif()
    endif()
  else()
    message(STATUS "team_policy: vendor_sort=OFF; forcing Kokkos::BinSort "
                   "fallback for spatial sort_by_key")
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

      # On Cray systems (e.g. Frontier) GPU-aware Cray MPICH can only
      # handle device pointers if the GPU Transport Layer (GTL) library
      # is linked. The Cray compiler wrappers (cc/CC) inject this
      # automatically, but we build with hipcc/nvcc directly, so
      # find_package(MPI) only finds base libmpi and the GTL is left
      # out -> MPI_Sendrecv on a device pointer fails with
      # "OFI ... Bad address". Add it explicitly here.
      #
      # Cray PE exports PE_MPICH_GTL_DIR_<accel> / PE_MPICH_GTL_LIBS_<accel>
      # (e.g. amd_gfx90a -> -lmpi_gtl_hsa). Their absence means this is
      # not a Cray MPICH build, in which case nothing extra is needed.
      if("${Kokkos_DEVICES}" MATCHES "HIP")
        set(_gtl_accels amd_gfx942 amd_gfx940 amd_gfx90a amd_gfx908 amd_gfx906)
      elseif("${Kokkos_DEVICES}" MATCHES "CUDA")
        set(_gtl_accels nvidia90 nvidia80 nvidia70)
      elseif("${Kokkos_DEVICES}" MATCHES "SYCL")
        set(_gtl_accels ponteVecchio)
      else()
        set(_gtl_accels "")
      endif()

      set(_gtl_dir "")
      set(_gtl_libflag "")
      foreach(_accel ${_gtl_accels})
        if((NOT _gtl_dir) AND (DEFINED ENV{PE_MPICH_GTL_DIR_${_accel}}))
          # strip the leading "-L" from the Cray-provided value
          string(REGEX REPLACE "^-L" ""
                 _gtl_dir "$ENV{PE_MPICH_GTL_DIR_${_accel}}")
          string(REGEX REPLACE "^-l" ""
                 _gtl_libflag "$ENV{PE_MPICH_GTL_LIBS_${_accel}}")
        endif()
      endforeach()

      if(_gtl_dir AND _gtl_libflag)
        find_library(MPI_GTL_LIBRARY
          NAMES ${_gtl_libflag}
          HINTS "${_gtl_dir}"
          NO_DEFAULT_PATH)
        if(MPI_GTL_LIBRARY)
          message(STATUS
            "GPU-aware MPI: linking Cray GTL library ${MPI_GTL_LIBRARY}")
          set(DEPENDENCIES ${DEPENDENCIES} ${MPI_GTL_LIBRARY})
        else()
          message(FATAL_ERROR
            "${Red}gpu_aware_mpi=ON: Cray MPICH detected but the GTL "
            "library 'lib${_gtl_libflag}' was not found in '${_gtl_dir}'. "
            "GPU-aware MPI will crash at runtime without it. Make sure the "
            "craype-accel module is loaded, or build with gpu_aware_mpi=OFF."
            "${ColorReset}")
        endif()
      else()
        message(STATUS
          "GPU-aware MPI: no Cray GTL environment found; assuming the MPI "
          "implementation is GPU-aware without an extra transport library.")
      endif()
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

# Ascent (in situ visualization)
if(${ascent})
  if(NOT ${output})
    message(
      FATAL_ERROR
      "${Red}Ascent requires `output=ON` to be enabled.${ColorReset}")
  endif()
  find_package(Ascent REQUIRED)
  add_compile_options("-D ASCENT_ENABLED")
  if(${mpi})
    set(DEPENDENCIES ${DEPENDENCIES} ascent::ascent_mpi)
  else()
    set(DEPENDENCIES ${DEPENDENCIES} ascent::ascent)
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
