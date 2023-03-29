set(kokkos_REPOSITORY https://github.com/kokkos/kokkos.git CACHE STRING "Kokkos repository")
set(fmt_REPOSITORY https://github.com/fmtlib/fmt.git CACHE STRING "fmt repository")
set(plog_REPOSITORY https://github.com/SergiusTheBest/plog.git CACHE STRING "plog repository")
set(toml11_REPOSITORY https://github.com/ToruNiina/toml11 CACHE STRING "toml11 repository")

# set (adios2_REPOSITORY https://github.com/ornladios/ADIOS2.git CACHE STRING "ADIOS2 repository")
function(find_or_fetch_dependency package_name header_only)
  if(NOT header_only)
    find_package(${package_name} QUIET)
  endif()

  if(NOT(${package_name}_FOUND))
    if(DEFINED ${package_name}_REPOSITORY)
      # fetching package
      message(STATUS "${Blue}${package_name} not found. Fetching from ${${package_name}_REPOSITORY}${ColorReset}")
      include(FetchContent)
      FetchContent_Declare(
        ${package_name}
        GIT_REPOSITORY ${${package_name}_REPOSITORY}
      )
      FetchContent_MakeAvailable(${package_name})
      set(${package_name}_ROOT ${CMAKE_CURRENT_BINARY_DIR}/_deps/${package_name}-build CACHE PATH "Path to ${package_name} build")
      set(${package_name}_SRC ${CMAKE_CURRENT_BINARY_DIR}/_deps/${package_name}-src CACHE PATH "Path to ${package_name} src")
      set(${package_name}_FETCHED TRUE CACHE BOOL "Whether ${package_name} was fetched")
      message(STATUS "${Green}${package_name} fetched.${ColorReset}")
    else()
      # get as submodule
      message(STATUS "${Yellow}${package_name} not found. Getting as submodule.${ColorReset}")
      execute_process(
        COMMAND git submodule update --init --remote ${CMAKE_CURRENT_SOURCE_DIR}/extern/${package_name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      )
      add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/${package_name} extern/${package_name})
      set(${package_name}_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/extern/${package_name} CACHE PATH "Path to ${package_name} root")
      set(${package_name}_SRC ${CMAKE_CURRENT_SOURCE_DIR}/extern/${package_name} CACHE PATH "Path to ${package_name} src")
    endif()
  else()
    message(STATUS "${Green}${package_name} found.${ColorReset}")
  endif()
endfunction()

if(${nttiny} STREQUAL "ON")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/nttiny extern/nttiny)
endif()

find_or_fetch_dependency(kokkos FALSE)
find_or_fetch_dependency(fmt FALSE)
find_or_fetch_dependency(plog TRUE)
find_or_fetch_dependency(toml11 TRUE)

list(APPEND DEPENDENCIES Kokkos::kokkos fmt::fmt)

if(${output} STREQUAL "ON")
  find_or_fetch_dependency(adios2 FALSE)
  list(APPEND DEPENDENCIES adios2::cxx11)
endif()

include_directories(${plog_SRC}/include)
include_directories(${toml11_SRC})

link_libraries(${DEPENDENCIES})
