cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME entity)

project(${PROJECT_NAME}
  VERSION 0.9.0.4
  LANGUAGES CXX C
)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/styling.cmake)

# ----------------------------- Defaults ---------------------------------- #
set(default_engine "pic")
set(default_precision "single")
set(default_pgen "dummy")
set(default_output "OFF")
set(default_DEBUG "OFF")
set(default_nttiny "OFF")
set(default_KOKKOS_ENABLE_CUDA "OFF")
set(default_KOKKOS_ENABLE_OPENMP "OFF")

# ----------------------------- Simulation engine ---------------------------- #
set(engine ${default_engine} CACHE STRING "Simulation engine")

set(simulation_engines "sandbox" "pic" "grpic")
list(FIND simulation_engines ${engine} ENGINE_FOUND)

if(${ENGINE_FOUND} EQUAL -1)
  message(FATAL_ERROR "Invalid simulation engine: ${engine}\nValid options are: ${simulation_engines}")
endif()

# -------------------------------- Precision ------------------------------- #
set(precision ${default_precision} CACHE STRING "Precision")
set(precisions "single" "double")
list(FIND precisions ${precision} PRECISION_FOUND)

if(${PRECISION_FOUND} EQUAL -1)
  message(FATAL_ERROR "Invalid precision: ${precision}\nValid options are: ${precisions}")
endif()

if(${precision} STREQUAL "single")
  add_definitions(-DSINGLE_PRECISION)
endif()

# --------------------------------- Metric --------------------------------- #
set(sr_metrics "minkowski" "spherical" "qspherical")
set(gr_metrics "kerr_schild" "qkerr_schild" "kerr_schild_0")

if(${engine} STREQUAL "sandbox")
  set(default_metric "minkowski")
  set(metrics ${sr_metrics} ${gr_metrics})
elseif(${engine} STREQUAL "pic")
  set(default_metric "minkowski")
  set(metrics ${sr_metrics})
elseif(${engine} STREQUAL "grpic")
  set(default_metric "qkerr_schild")
  set(metrics ${gr_metrics})
endif()

set(metric ${default_metric} CACHE STRING "Metric")

list(FIND metrics ${metric} METRIC_FOUND)

if(${METRIC_FOUND} EQUAL -1)
  message(FATAL_ERROR "Invalid metric: ${metric}\nValid options are: ${metrics}")
endif()

# ---------------------------- Problem generator --------------------------- #
if(NOT ${engine} STREQUAL "sandbox")
  set(PGEN_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/${engine}/pgen/)
  set(pgen ${default_pgen} CACHE STRING "Problem generator")
  file(GLOB_RECURSE PGENS ${PGEN_DIRECTORY}*.hpp)
  set(problem_generators "")

  foreach(pgen_file ${PGENS})
    string(REPLACE ${PGEN_DIRECTORY} "" new_pgen ${pgen_file})
    string(REPLACE ".hpp" "" new_pgen ${new_pgen})
    list(APPEND problem_generators ${new_pgen})
  endforeach()

  list(FIND problem_generators ${pgen} PGEN_FOUND)

  if(${PGEN_FOUND} EQUAL -1)
    message(FATAL_ERROR "Problem generator ${pgen}.hpp not found\nAvailable problem generators: ${problem_generators}.")
  endif()
endif()

# -------------------------- Compilation settings -------------------------- #
set(DEBUG ${default_DEBUG} CACHE BOOL "Debug mode")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(${DEBUG} STREQUAL "OFF")
  set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL "Kokkos aggressive vectorization")
  set(Kokkos_ENABLE_COMPILER_WARNINGS OFF CACHE BOOL "Kokkos compiler warnings")
  set(Kokkos_ENABLE_DEBUG OFF CACHE BOOL "Kokkos debug")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK OFF CACHE BOOL "Kokkos debug bounds check")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "CMake build type")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
  message(STATUS "Release mode")
else()
  set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION OFF CACHE BOOL "Kokkos aggressive vectorization")
  set(Kokkos_ENABLE_COMPILER_WARNINGS ON CACHE BOOL "Kokkos compiler warnings")
  set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Kokkos debug")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL "Kokkos debug bounds check")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "CMake build type")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  message(STATUS "Debug mode")
endif()

# ---------------------------------- Tests --------------------------------- #
set(BUILD_TESTING OFF CACHE BOOL "Build tests")

if(${BUILD_TESTING} STREQUAL "OFF")
  set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Kokkos tests")
else()
  set(Kokkos_ENABLE_TESTS ON CACHE BOOL "Kokkos tests")
endif()

# ----------------------------------- GUI ---------------------------------- #
set(nttiny ${default_nttiny} CACHE BOOL "Use nttiny GUI")

# ----------------------------- Kokkos settings ---------------------------- #
set(Kokkos_ENABLE_CUDA ${default_KOKKOS_ENABLE_CUDA} CACHE BOOL "Enable CUDA")
set(Kokkos_ENABLE_OPENMP ${default_KOKKOS_ENABLE_OPENMP} CACHE BOOL "Enable OpenMP")

if(${Kokkos_ENABLE_CUDA})
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable CUDA lambda")
endif()

# set memory space
if(${Kokkos_ENABLE_CUDA})
  add_definitions(-DENABLE_GPU)
  set(ACC_MEM_SPACE Kokkos::CudaSpace)
else()
  set(ACC_MEM_SPACE Kokkos::HostSpace)
endif()

set(HOST_MEM_SPACE Kokkos::HostSpace)

# set execution space
if(${Kokkos_ENABLE_CUDA})
  set(ACC_EXE_SPACE Kokkos::Cuda)
elseif(${Kokkos_ENABLE_OPENMP})
  set(ACC_EXE_SPACE Kokkos::OpenMP)
else()
  set(ACC_EXE_SPACE Kokkos::Serial)
endif()

if(${Kokkos_ENABLE_OPENMP})
  set(HOST_EXE_SPACE Kokkos::OpenMP)
else()
  set(HOST_EXE_SPACE Kokkos::Serial)
endif()

add_compile_options("-D AccelExeSpace=${ACC_EXE_SPACE}")
add_compile_options("-D AccelMemSpace=${ACC_MEM_SPACE}")
add_compile_options("-D HostExeSpace=${HOST_EXE_SPACE}")
add_compile_options("-D HostMemSpace=${HOST_MEM_SPACE}")

# ----------------------------- Adios2 settings ---------------------------- #
set(output ${default_output} CACHE BOOL "Enable output")

if(${output} STREQUAL "ON")
  set(ADIOS2_BUILD_EXAMPLES OFF CACHE BOOL "Build ADIOS2 examples")

  # Language support
  set(ADIOS2_USE_Python OFF CACHE BOOL "Use Python for ADIOS2")
  set(ADIOS2_USE_Fortran OFF CACHE BOOL "Use Fortran for ADIOS2")

  # Format/compression support
  set(ADIOS2_USE_ZeroMQ OFF CACHE BOOL "Use ZeroMQ for ADIOS2")
  set(ADIOS2_USE_SST OFF CACHE BOOL "Use SST for ADIOS2")
  set(ADIOS2_USE_BZip2 OFF CACHE BOOL "Use BZip2 for ADIOS2")
  set(ADIOS2_USE_ZFP OFF CACHE BOOL "Use ZFP for ADIOS2")
  set(ADIOS2_USE_SZ OFF CACHE BOOL "Use SZ for ADIOS2")
  set(ADIOS2_USE_MGARD OFF CACHE BOOL "Use MGARD for ADIOS2")
  set(ADIOS2_USE_PNG OFF CACHE BOOL "Use PNG for ADIOS2")
  set(ADIOS2_USE_Blosc OFF CACHE BOOL "Use Blosc for ADIOS2")

  # !TODO: add MPI-enabled ADIOS2
  set(ADIOS2_USE_MPI OFF CACHE BOOL "Use MPI for ADIOS2")

  # !TODO: add CUDA-enabled ADIOS2
  set(ADIOS2_USE_CUDA OFF CACHE BOOL "Use CUDA for ADIOS2")

  # set(ADIOS2_USE_CUDA ${Kokkos_ENABLE_CUDA} CACHE BOOL "Use CUDA for ADIOS2")
  add_compile_options("-D OUTPUT_ENABLED")
endif()

# ------------------------------ Dependencies ------------------------------ #
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies.cmake)

# ------------------------------- Main source ------------------------------ #
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src src)

# ---------------------------------- Tests --------------------------------- #
include(CTest)

if(TESTS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests tests)
endif(TESTS)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/report.cmake)