cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME entity)

project(${PROJECT_NAME}
        VERSION 0.8.0
        LANGUAGES CXX
)

# ----------------------------- Simulation type ---------------------------- #
set(simtype "pic" CACHE STRING "Simulation type")
set(simulation_types "pic" "grpic")
list(FIND simulation_types ${simtype} SIMTYPE_FOUND)
if(${SIMTYPE_FOUND} EQUAL -1)
  message(FATAL_ERROR "Invalid simulation type: ${simtype}\nValid options are: ${simulation_types}")
endif()

# --------------------------------- Metric --------------------------------- #
if (${simtype} STREQUAL "pic")
  set(metric "minkowski" CACHE STRING "Metric")
  set(metrics "minkowski" "spherical" "qspherical")
elseif (${simtype} STREQUAL "grpic")
  set(metric "spherical" CACHE STRING "Metric")
  set(metrics "kerr_schild" "qkerr_schild")
endif()
list(FIND metrics ${metric} METRIC_FOUND)
if(${METRIC_FOUND} EQUAL -1)
  message(FATAL_ERROR "Invalid metric: ${metric}\nValid options are: ${metrics}")
endif()

# ---------------------------- Problem generator --------------------------- #
# !TODO: add empty pgen
set(PGEN_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/${simtype}/pgen/)
set(pgen "dummy" CACHE STRING "Problem generator")
file(GLOB_RECURSE PGENS ${PGEN_DIRECTORY}*.cpp)
set(problem_generators "")
foreach(pgen_file ${PGENS})
  string(REPLACE ${PGEN_DIRECTORY} "" new_pgen ${pgen_file})
  list(APPEND problem_generators ${new_pgen})
endforeach()
list(FIND problem_generators ${pgen}.cpp PGEN_FOUND)
if (${PGEN_FOUND} EQUAL -1)
  message(FATAL_ERROR "Problem generator ${pgen}.cpp not found\nAvailable problem generators: ${problem_generators}")
endif()
set(PGEN_SRC ${PGEN_DIRECTORY}${pgen}.cpp)

# -------------------------- Compilation settings -------------------------- #
set(DEBUG OFF CACHE BOOL "Debug mode")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (${DEBUG} STREQUAL "OFF")
  set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON)
  set(Kokkos_ENABLE_COMPILER_WARNINGS OFF)
  set(Kokkos_ENABLE_DEBUG OFF)
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK OFF)
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -DNDEBUG")
  message(STATUS "Release mode")
else()
  set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION OFF)
  set(Kokkos_ENABLE_COMPILER_WARNINGS ON)
  set(Kokkos_ENABLE_DEBUG ON)
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON)
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -DDEBUG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  message(STATUS "Debug mode")
endif()

# ----------------------------------- GUI ---------------------------------- #
set(nttiny OFF CACHE BOOL "Use nttiny GUI")
if (${nttiny} STREQUAL "ON")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/nttiny nttiny)
endif()

# ----------------------------- Kokkos settings ---------------------------- #
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/kokkos extern/kokkos)

# set memory space
if(${Kokkos_ENABLE_CUDA})
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
set(ENABLE_OUTPUT ON CACHE BOOL "Enable output")
if (${ENABLE_OUTPUT} STREQUAL "ON")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/adios2 extern/adios2)
  set(BUILD_TESTING OFF CACHE BOOL "Build adios2 tests")
endif()

# ------------------------------- Main source ------------------------------ #
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src src)

include(${CMAKE_CURRENT_SOURCE_DIR}/report.cmake)