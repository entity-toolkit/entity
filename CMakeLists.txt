cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0110 NEW)

set(PROJECT_NAME entity)

project(${PROJECT_NAME}
  VERSION 0.9.0.5
  LANGUAGES CXX C
)
add_compile_options("-D ENTITY_VERSION=\"${PROJECT_VERSION}\"")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/styling.cmake)

# -------------------------- Compilation settings -------------------------- #
set(DEBUG OFF CACHE BOOL "Debug mode")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(${DEBUG} STREQUAL "OFF")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "CMake build type")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
  message(STATUS "Release mode")
else()
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "CMake build type")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG -Wall -Wextra")
  message(STATUS "Debug mode")
endif()

# ----------------------------- Configurations ----------------------------- #
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/defaults.cmake)

# defaults
set(engine ${default_engine} CACHE STRING "Simulation engine")
set(precision ${default_precision} CACHE STRING "Precision")
set(pgen ${default_pgen} CACHE STRING "Problem generator")

set(nttiny ${default_nttiny} CACHE BOOL "Use nttiny GUI")
set(output ${default_output} CACHE BOOL "Enable output")

# options
set(simulation_engines "sandbox" "pic" "grpic" CACHE STRING "Simulation engines")
set(precisions "single" "double" CACHE STRING "Precisions")
set(sr_metrics "minkowski" "spherical" "qspherical" CACHE STRING "SR metrics")
set(gr_metrics "kerr_schild" "qkerr_schild" "kerr_schild_0" CACHE STRING "GR metrics")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)

# ------------------------- Third-Party Tests ------------------------------ #
set(BUILD_TESTING OFF CACHE BOOL "Build tests")

# ------------------------------ Dependencies ------------------------------ #
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/kokkosConfig.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies.cmake)

find_or_fetch_dependency(Kokkos FALSE)
find_or_fetch_dependency(fmt TRUE)
find_or_fetch_dependency(plog TRUE)
find_or_fetch_dependency(toml11 TRUE)
set(DEPENDENCIES Kokkos::kokkos fmt::fmt-header-only)
include_directories(${plog_SRC}/include)
include_directories(${toml11_SRC})

# -------------------------------- Main code ------------------------------- #
set_precision(${precision})

if(TESTS)
  # ---------------------------------- Tests --------------------------------- #
  link_libraries(${DEPENDENCIES})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests tests)
else()
  # ----------------------------------- GUI ---------------------------------- #
  if(${nttiny})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/nttiny extern/nttiny)
  endif()

  # --------------------------------- Output --------------------------------- #
  if(${output})
    message("Kokkos_ROOT: ${Kokkos_ROOT}")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/adios2Config.cmake)
    find_or_fetch_dependency(adios2 FALSE)
    set(DEPENDENCIES ${DEPENDENCIES} adios2::cxx11)
  endif()

  # ------------------------------- Main source ------------------------------ #
  link_libraries(${DEPENDENCIES})
  set_engine(${engine})
  set(metric ${default_metric} CACHE STRING "Metric")
  set_metric(${metric} ${engine})
  set_problem_generator(${pgen} ${engine})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src src)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/report.cmake)
endif(TESTS)