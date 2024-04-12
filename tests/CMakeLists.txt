include(CTest)
enable_testing()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

# add_subdirectory(${SRC_DIR}/global ${CMAKE_CURRENT_BINARY_DIR}/global)

add_subdirectory(${SRC_DIR}/global/tests ${CMAKE_CURRENT_BINARY_DIR}/global)
add_subdirectory(${SRC_DIR}/metrics/tests ${CMAKE_CURRENT_BINARY_DIR}/metrics)
add_subdirectory(${SRC_DIR}/kernels/tests ${CMAKE_CURRENT_BINARY_DIR}/kernels)
add_subdirectory(${SRC_DIR}/archetypes/tests ${CMAKE_CURRENT_BINARY_DIR}/archetypes)
add_subdirectory(${SRC_DIR}/framework/tests ${CMAKE_CURRENT_BINARY_DIR}/framework)

if (${output} STREQUAL "ON")
add_subdirectory(${SRC_DIR}/output/tests ${CMAKE_CURRENT_BINARY_DIR}/output)
endif()