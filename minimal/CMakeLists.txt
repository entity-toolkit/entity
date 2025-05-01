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
    "ADIOS2_NOMPI;ADIOS2_MPI"
    CACHE STRING "Build modes")

if("ADIOS2_NOMPI" IN_LIST MODES)
  include(adios2-nompi.cmake)
endif()

if("ADIOS2_MPI" IN_LIST MODES)
  include(adios2-mpi.cmake)
endif()

if("MPI" IN_LIST MODES)
  include(mpi.cmake)
endif()

if("MPI_SIMPLE" IN_LIST MODES)
  include(mpi-simple.cmake)
endif()

message(STATUS "Build modes: ${MODES}")
