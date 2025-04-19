include(CTest)
enable_testing()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_subdirectory(${SRC_DIR}/global ${CMAKE_CURRENT_BINARY_DIR}/global)
add_subdirectory(${SRC_DIR}/metrics ${CMAKE_CURRENT_BINARY_DIR}/metrics)
add_subdirectory(${SRC_DIR}/kernels ${CMAKE_CURRENT_BINARY_DIR}/kernels)
add_subdirectory(${SRC_DIR}/archetypes ${CMAKE_CURRENT_BINARY_DIR}/archetypes)
add_subdirectory(${SRC_DIR}/framework ${CMAKE_CURRENT_BINARY_DIR}/framework)
add_subdirectory(${SRC_DIR}/output ${CMAKE_CURRENT_BINARY_DIR}/output)
if(${output})
  add_subdirectory(${SRC_DIR}/checkpoint ${CMAKE_CURRENT_BINARY_DIR}/checkpoint)
endif()

set(TEST_DIRECTORIES "")

if(NOT ${mpi})
  list(APPEND TEST_DIRECTORIES global)
  list(APPEND TEST_DIRECTORIES metrics)
  list(APPEND TEST_DIRECTORIES kernels)
  list(APPEND TEST_DIRECTORIES archetypes)
  list(APPEND TEST_DIRECTORIES framework)
elseif(${mpi} AND ${output})
  list(APPEND TEST_DIRECTORIES framework)
endif()

list(APPEND TEST_DIRECTORIES output)

if(${output})
  list(APPEND TEST_DIRECTORIES checkpoint)
endif()

foreach(test_dir IN LISTS TEST_DIRECTORIES)
  add_subdirectory(${SRC_DIR}/${test_dir}/tests
                   ${CMAKE_CURRENT_BINARY_DIR}/${test_dir}/tests)
endforeach()
