file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

target_include_directories(${SIMENGINE} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(${SIMENGINE} PRIVATE ${SRC_FILES})
target_link_libraries(${SIMENGINE} PRIVATE ${FRAMEWORK} ${WRAPPER})

if(${nttiny} STREQUAL "ON")
  target_link_libraries(${SIMENGINE} PUBLIC nttiny)
endif()
