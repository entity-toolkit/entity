# cmake-lint: disable=C0103

# -------------------------------- Precision ------------------------------- #
function(set_precision precision_name)
  list(FIND precisions ${precision_name} PRECISION_FOUND)

  if(${PRECISION_FOUND} EQUAL -1)
    message(
      FATAL_ERROR
        "Invalid precision: ${precision_name}\nValid options are: ${precisions}"
    )
  endif()

  if(${precision_name} STREQUAL "single")
    add_compile_options("-DSINGLE_PRECISION")
  endif()
endfunction()

# ---------------------------- Problem generator --------------------------- #
function(set_problem_generator pgen_name)
  if(pgen_name STREQUAL ".")
    message(FATAL_ERROR "Problem generator not specified")
  endif()

  file(GLOB_RECURSE PGENS "${CMAKE_CURRENT_SOURCE_DIR}/pgens/**/pgen.hpp")

  foreach(PGEN ${PGENS})
    get_filename_component(PGEN_NAME ${PGEN} DIRECTORY)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/pgens/" "" PGEN_NAME
                   ${PGEN_NAME})
    list(APPEND PGEN_NAMES ${PGEN_NAME})
  endforeach()

  list(FIND PGEN_NAMES ${pgen_name} PGEN_FOUND)

  file(GLOB_RECURSE EXTRA_PGENS
       "${CMAKE_CURRENT_SOURCE_DIR}/extern/entity-pgens/**/pgen.hpp")
  foreach(EXTRA_PGEN ${EXTRA_PGENS})
    get_filename_component(EXTRA_PGEN_NAME ${EXTRA_PGEN} DIRECTORY)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/extern/entity-pgens/" ""
                   EXTRA_PGEN_NAME ${EXTRA_PGEN_NAME})
    list(APPEND PGEN_NAMES "pgens/${EXTRA_PGEN_NAME}")
  endforeach()

  if(${PGEN_FOUND} EQUAL -1)
    if(${pgen_name} MATCHES "^pgens/")
      get_filename_component(pgen_name ${pgen_name} NAME)
      set(pgen_path
          "${CMAKE_CURRENT_SOURCE_DIR}/extern/entity-pgens/${pgen_name}")
      set(pgen_name "pgens/${pgen_name}")
    else()
      set(pgen_path ${pgen_name})
      get_filename_component(pgen_path ${pgen_path} ABSOLUTE)
      string(REGEX REPLACE ".*/" "" pgen_name ${pgen_name})
      list(APPEND PGEN_NAMES ${pgen_name})
    endif()
  else()
    set(pgen_path ${CMAKE_CURRENT_SOURCE_DIR}/pgens/${pgen_name})
  endif()

  file(GLOB_RECURSE PGEN_FILES "${pgen_path}/pgen.hpp")
  if(NOT PGEN_FILES)
    message(FATAL_ERROR "pgen.hpp file not found in ${pgen_path}")
  endif()

  add_library(ntt_pgen INTERFACE)
  target_link_libraries(ntt_pgen INTERFACE ntt_global ntt_framework
                                           ntt_archetypes ntt_kernels)

  target_include_directories(ntt_pgen INTERFACE ${pgen_path})

  set(PGEN
      ${pgen_name}
      PARENT_SCOPE)
  set(PGEN_FOUND
      TRUE
      PARENT_SCOPE)
  set(problem_generators
      ${PGEN_NAMES}
      PARENT_SCOPE)
endfunction()
