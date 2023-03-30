set(WRAPPER ${PROJECT_NAME}-wrapper)
set(FRAMEWORK ${PROJECT_NAME}-framework)
set(SIMENGINE ${PROJECT_NAME}-${engine})

set(ENTITY ${PROJECT_NAME}.xc)
set(VIS nttiny.xc)

set(EXTERN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../extern)

# configuration file
set(ENGINE_FLAG ${engine}_engine)
string(TOUPPER ${ENGINE_FLAG} ENGINE_FLAG)
set(METRIC_FLAG ${metric}_metric)
string(TOUPPER ${METRIC_FLAG} METRIC_FLAG)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/config.h"
)

# wrapper file
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/wrapper/kokkos.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/wrapper.h"
)

# problem generator
if(NOT ${PGEN_FOUND} EQUAL -1)
  configure_file(
    "${PGEN_DIRECTORY}/${pgen}.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/problem_generator.hpp"
  )
endif()

# metric file
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/framework/metrics/${metric}.h"
  "${CMAKE_CURRENT_BINARY_DIR}/metric.h"
)

add_library(${FRAMEWORK} STATIC)
add_library(${SIMENGINE} STATIC)
add_library(${WRAPPER} STATIC)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/framework)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/framework/meshblock)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/framework/io)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/framework/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/framework/vec_transforms)
include_directories(${EXTERN_DIR})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/wrapper wrapper)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/framework framework)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${engine} ${engine})

add_executable(${ENTITY} entity.cpp)
target_link_libraries(${ENTITY} PUBLIC ${WRAPPER} ${FRAMEWORK} ${SIMENGINE})
target_include_directories(${ENTITY} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/${engine})

if(${nttiny} STREQUAL "ON")
  add_compile_options("-D NTTINY_ENABLED")
  add_executable(${VIS} nttiny.cpp)
  target_link_libraries(${VIS} PRIVATE nttiny ${WRAPPER} ${FRAMEWORK} ${SIMENGINE})
  target_include_directories(${VIS} PRIVATE ${EXTERN_DIR}/nttiny/ ${CMAKE_CURRENT_SOURCE_DIR}/${engine}/)
endif()