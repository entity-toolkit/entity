# Configure
set(ENTITY ${PROJECT_NAME}.xc)

set(EXTERN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../extern)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/wrapper)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/framework)
include_directories(${EXTERN_DIR})

# Wrapper
set(WRAPPER ${PROJECT_NAME}-wrapper)
add_library(${WRAPPER} STATIC ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/kokkos.cpp)
link_libraries(${WRAPPER})

# Framework
set(FRAMEWORK ${PROJECT_NAME}-framework)
file(GLOB_RECURSE FRAMEWORK_SRC_FILES
     ${CMAKE_CURRENT_SOURCE_DIR}/framework/*.cpp)
add_library(${FRAMEWORK} STATIC ${FRAMEWORK_SRC_FILES})
link_libraries(${FRAMEWORK})

# Engine
set(SIMENGINE ${PROJECT_NAME}-${engine})
file(GLOB_RECURSE SIMENGINE_SRC_FILES
     ${CMAKE_CURRENT_SOURCE_DIR}/engines/${engine}/*.cpp)
add_library(${SIMENGINE} STATIC ${SIMENGINE_SRC_FILES})

target_include_directories(
  ${SIMENGINE} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/engines
                      ${CMAKE_CURRENT_SOURCE_DIR}/engines/${engine})

# Main app
add_executable(${ENTITY} entity.cpp)

target_link_libraries(${ENTITY} PUBLIC ${SIMENGINE})
target_include_directories(${ENTITY}
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/engines/${engine})

if(${gui})
  set(VIS ${PROJECT_NAME}-GUI.xc)
  add_compile_options("-D GUI_ENABLED")
  add_executable(${VIS} nttiny.cpp)
  target_link_libraries(${VIS} PRIVATE nttiny ${SIMENGINE})
  target_include_directories(
    ${VIS} PRIVATE ${EXTERN_DIR}/nttiny/
                   ${CMAKE_CURRENT_SOURCE_DIR}/engines/${engine}/)
  target_link_libraries(${SIMENGINE} PUBLIC nttiny)
endif()
