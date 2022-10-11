file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set_property(TARGET ${FRAMEWORK} PROPERTY CXX_STANDARD 17)
target_sources(${FRAMEWORK} PRIVATE ${SRC_FILES})
target_link_libraries(${FRAMEWORK} PRIVATE Kokkos::kokkos ${WRAPPER})

if(${output} STREQUAL "ON")
  target_link_libraries(${FRAMEWORK} PRIVATE adios2::cxx11)
endif()