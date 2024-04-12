# ------------------------------
# @brief: Generates tests for the `ntt_archetypes` module
# @uses:
# - kokkos [required]
# - plog [required]
# - mpi [optional]
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

add_subdirectory(${SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/archetypes)

function(gen_test title)
  set(exec test-archetypes-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  target_link_libraries(${exec} PRIVATE ntt_archetypes ntt_global ntt_metrics)

  add_test(NAME "ARCHETYPES::${title}" COMMAND "${exec}")
endfunction()

gen_test(energy_dist)
gen_test(spatial_dist)
gen_test(field_setter)