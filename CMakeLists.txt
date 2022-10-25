cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME entity)

project(${PROJECT_NAME}
  VERSION 0.8.0.5
  LANGUAGES CXX
)

find_package(Git QUIET)

if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  option(GIT_SUBMODULE "Check submodules during build" ON)

  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)

    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
  endif()
endif()

set(default_simtype "pic")
set(default_precision "single")
set(default_pgen "dummy")
set(default_output "ON")
set(default_DEBUG "OFF")
set(default_nttiny "OFF")
set(default_KOKKOS_ENABLE_CUDA "OFF")
set(default_KOKKOS_ENABLE_OPENMP "OFF")

# ----------------------------- Simulation type ---------------------------- #
set(simtype ${default_simtype} CACHE STRING "Simulation type")

if(simtype STREQUAL "grpic")
  message(FATAL_ERROR "GRPIC is not supported yet. See the `dev/gr` branch.")
endif()

set(simulation_types "pic" "grpic")
list(FIND simulation_types ${simtype} SIMTYPE_FOUND)

if(${SIMTYPE_FOUND} EQUAL -1)
  message(FATAL_ERROR "Invalid simulation type: ${simtype}\nValid options are: ${simulation_types}")
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
if(${simtype} STREQUAL "pic")
  set(default_metric "minkowski")
  set(metric ${default_metric} CACHE STRING "Metric")
  set(metrics "minkowski" "spherical" "qspherical")
elseif(${simtype} STREQUAL "grpic")
  set(default_metric "qkerr_schild")
  set(metric ${default_metric} CACHE STRING "Metric")
  set(metrics "kerr_schild" "qkerr_schild")
endif()

list(FIND metrics ${metric} METRIC_FOUND)

if(${METRIC_FOUND} EQUAL -1)
  message(FATAL_ERROR "Invalid metric: ${metric}\nValid options are: ${metrics}")
endif()

# ---------------------------- Problem generator --------------------------- #
set(PGEN_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/${simtype}/pgen/)
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
  message(FATAL_ERROR "Problem generator ${pgen}.hpp not found\nAvailable problem generators: ${problem_generators}")
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

# ---------------------------------- tests --------------------------------- #
set(BUILD_TESTING OFF CACHE BOOL "Build tests")

if(${BUILD_TESTING} STREQUAL "OFF")
  set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Kokkos tests")
else()
  set(Kokkos_ENABLE_TESTS ON CACHE BOOL "Kokkos tests")
endif()

# ----------------------------------- GUI ---------------------------------- #
set(nttiny ${default_nttiny} CACHE BOOL "Use nttiny GUI")

if(${nttiny} STREQUAL "ON")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../nttiny extern/nttiny)
endif()

# ----------------------------- Kokkos settings ---------------------------- #
if(${Kokkos_ENABLE_CUDA})
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable CUDA lambda")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/kokkos extern/kokkos)

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

# ----------------------------- Adios2 settings ---------------------------- #
set(output ${default_output} CACHE BOOL "Enable output")

if(${output} STREQUAL "ON")
  set(ADIOS2_USE_Python OFF CACHE BOOL "Use Python for ADIOS2")
  set(ADIOS2_USE_Fortran OFF CACHE BOOL "Use Fortran for ADIOS2")
  set(ADIOS2_USE_MPI OFF CACHE BOOL "Use MPI for ADIOS2")
  set(ADIOS2_BUILD_EXAMPLES OFF CACHE BOOL "Build ADIOS2 examples")
  add_compile_options("-D OUTPUT_ENABLED")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/adios2 extern/adios2)
endif()

# ----------------------------------- fmt ---------------------------------- #
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/fmt extern/fmt)

# ------------------------------- Main source ------------------------------ #
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src src)

include(${CMAKE_CURRENT_SOURCE_DIR}/report.cmake)