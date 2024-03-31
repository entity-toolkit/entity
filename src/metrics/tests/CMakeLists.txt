# ------------------------------
# @brief: Generates tests for the `ntt_metrics` module
# @uses:
# - kokkos [required]
# - plog [required]
# - mpi [optional]
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

add_subdirectory(${SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/metrics)

function(gen_test title)
  set(exec test-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  target_link_libraries(${exec} PRIVATE ntt_metrics)

  add_test(NAME "METRICS::${title}" COMMAND "${exec}")
endfunction()

gen_test(minkowski)
gen_test(vec_trans)