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

function(gen_test title)
  set(exec test-output-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  set (libs ntt_output ntt_global ntt_metrics ntt_framework)
  add_dependencies(${exec} ${libs})
  target_link_libraries(${exec} PRIVATE ${libs})

  add_test(NAME "OUTPUT::${title}" COMMAND "${exec}")
endfunction()

if (NOT ${mpi})
  gen_test(fields)
  gen_test(writer-nompi)
else()
  gen_test(writer-mpi)
endif()