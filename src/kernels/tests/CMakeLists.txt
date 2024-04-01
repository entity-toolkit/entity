# ------------------------------
# @brief: Generates tests for the `ntt_kernels` module
# @uses:
# - kokkos [required]
# - plog [required]
# - mpi [optional]
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

add_subdirectory(${SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/kernels)

function(gen_test title)
  set(exec test-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  target_link_libraries(${exec} PRIVATE ntt_kernels ntt_metrics)

  add_test(NAME "KERNELS::${title}" COMMAND "${exec}")
endfunction()

gen_test(faraday-mink)
gen_test(ampere-mink)