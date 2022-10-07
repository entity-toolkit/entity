file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set_property(TARGET ${FRAMEWORK} PROPERTY CXX_STANDARD 17)

# target_include_directories(${FRAMEWORK} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# target_include_directories(${FRAMEWORK} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/io)
# target_include_directories(${FRAMEWORK} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils)
# target_include_directories(${FRAMEWORK} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vec_transforms)
target_sources(${FRAMEWORK} PRIVATE ${SRC_FILES})
target_link_libraries(${FRAMEWORK} PRIVATE Kokkos::kokkos ${WRAPPER} stdc++fs)
