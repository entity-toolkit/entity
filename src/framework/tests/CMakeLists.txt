# ------------------------------
# @brief: Generates tests for the `ntt_framework` module
# @uses:
# - kokkos [required]
# - plog [required]
# - toml11 [required]
# - mpi [optional]
# - adios2 [optional]
# !TODO:
# - add tests for mesh separately
# - add test for 3D metadomain
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

add_subdirectory(${SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/framework)

function(gen_test title)
  set(exec test-framework-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  target_link_libraries(${exec} PRIVATE ntt_framework ntt_global ntt_metrics)

  add_test(NAME "FRAMEWORK::${title}" COMMAND "${exec}")
endfunction()

gen_test(parameters)
gen_test(particles)
gen_test(fields)
gen_test(grid_mesh)
gen_test(metadomain)

# this test is only run manually to ensure ...
# ... command line args are working properly ...
# ... and that the logging is done correctly
# gen_test(simulation) 