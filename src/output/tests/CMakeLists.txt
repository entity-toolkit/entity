# ------------------------------
# @brief: Generates tests for the `ntt_output` module
# @uses:
# - kokkos [required]
# - mpi [optional]
# - adios2 [optional]
# !TODO:
# - add more proper write tests for ADIOS2
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

add_subdirectory(${SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/output)

function(gen_test title)
  set(exec test-output-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  target_link_libraries(${exec} PRIVATE ntt_output ntt_global ntt_metrics ntt_framework)

  add_test(NAME "OUTPUT::${title}" COMMAND "${exec}")
endfunction()

gen_test(fields)

if (NOT ${mpi})
gen_test(writer-nompi)
endif()