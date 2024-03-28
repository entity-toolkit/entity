include(CTest)
enable_testing()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

add_subdirectory(${SRC_DIR}/global/tests ${CMAKE_CURRENT_BINARY_DIR}/global)