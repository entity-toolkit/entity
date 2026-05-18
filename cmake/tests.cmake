include(CTest)
enable_testing()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)

set(TEST_DIRECTORIES "")

list(APPEND TEST_DIRECTORIES global)
list(APPEND TEST_DIRECTORIES metrics)
list(APPEND TEST_DIRECTORIES kernels)
list(APPEND TEST_DIRECTORIES archetypes)
list(APPEND TEST_DIRECTORIES framework)
list(APPEND TEST_DIRECTORIES output)

foreach(test_dir IN LISTS TEST_DIRECTORIES)
  add_subdirectory(${SRC_DIR}/${test_dir}
                   ${CMAKE_CURRENT_BINARY_DIR}/tests/${test_dir})
endforeach()
