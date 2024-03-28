# ------------------------------
# @brief: Generates tests for the `global` module
# @uses:
# - kokkos [required]
# - plog [required]
# - mpi [optional]
# !TODO:
# - add optional tests for the `mpi_aliases.h`
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

add_subdirectory(${SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/global)

function(gen_test title)
  set(exec test-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  target_link_libraries(${exec} PRIVATE ntt_global)

  add_test(NAME "GLOBAL::${title}" COMMAND "${exec}")
endfunction()

gen_test(global)
gen_test(enums)
gen_test(kokkos_aliases)
gen_test(directions)
gen_test(comparators)
gen_test(numeric)
gen_test(param_container)
gen_test(sorting)
