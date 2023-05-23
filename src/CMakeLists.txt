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
set(SIMULATION_METRIC ${metric})

add_compile_options("-D ${ENGINE_FLAG}")
add_compile_options("-D ${METRIC_FLAG}")
add_compile_options("-D SIMULATION_METRIC=\"${SIMULATION_METRIC}\"")
add_compile_options("-D METRIC_HEADER=\"metrics/${metric}.h\"")
add_compile_options("-D ENTITY_VERSION=\"${PROJECT_VERSION}\"")

# problem generator
if(NOT ${PGEN_FOUND} EQUAL -1)
  configure_file(
    "${PGEN_DIRECTORY}/${pgen}.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/problem_generator.hpp"
  )
endif()

add_library(${FRAMEWORK} STATIC)
add_library(${SIMENGINE} STATIC)
add_library(${WRAPPER} STATIC)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/wrapper)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/framework)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/framework/meshblock)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/framework/io)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/framework/utils)
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