# ------------------------------
# @brief: Generates tests for the `ntt_framework` module
# @uses:
# - kokkos [required]
# - plog [required]
# - toml11 [required]
# - mpi [optional]
# - adios2 [optional]
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

add_subdirectory(${SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/framework)

function(gen_test title)
  set(exec test-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  target_link_libraries(${exec} PRIVATE ntt_framework ntt_global ntt_metrics)

  add_test(NAME "FRAMEWORK::${title}" COMMAND "${exec}")
endfunction()

gen_test(parameters)