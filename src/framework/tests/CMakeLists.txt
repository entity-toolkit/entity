# cmake-lint: disable=C0103,C0111
# ------------------------------
# @brief: Generates tests for the `ntt_framework` module
#
# @uses:
#
# * kokkos [required]
# * plog [required]
# * mpi [optional]
# * adios2 [optional]
#
# !TODO:
#
# * add tests for mesh separately
# * add test for 3D metadomain
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

function(gen_test title is_parallel)
  set(exec test-framework-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  set(libs ntt_framework ntt_global ntt_metrics)
  add_dependencies(${exec} ${libs})
  target_link_libraries(${exec} PRIVATE ${libs})

  if(${is_parallel})
    add_test(NAME "FRAMEWORK::${title}"
             COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "2"
                     "${exec}")
  else()
    add_test(NAME "FRAMEWORK::${title}" COMMAND "${exec}")
  endif()
endfunction()

gen_test(parameters false)
if(${mpi})
  gen_test(particles true)
  gen_test(particles_sort true)
else()
  gen_test(particles false)
  gen_test(particles_sort false)
endif()

gen_test(fields false)
gen_test(grid_mesh false)
if(${DEBUG})
  gen_test(metadomain false)
endif()

if(${mpi})
  gen_test(comm-mpi true)
else()
  gen_test(comm-nompi false)
endif()

# this test is only run manually to ensure ...
# ... command line args are working properly ...
# ... and that the logging is done correctly
#
# gen_test(simulation)
