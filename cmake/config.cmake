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

# ------------------------------- Shape function --------------------------- #
function(set_shape_order shape_order)
  if(${deposit} STREQUAL "esirkepov")
    if(${shape_order} GREATER 11)
      message(FATAL_ERROR "Shape order must be between 1 and 11.")
    endif()
    add_compile_options("-DSHAPE_ORDER=${shape_order}")
  endif()
endfunction()

# ---------------------------- Problem generator --------------------------- #
function(get_available_pgens available_pgens)
  set(available_pgens "")

  file(GLOB_RECURSE BASE_PGENS "${CMAKE_CURRENT_SOURCE_DIR}/pgens/**/pgen.hpp")

  foreach(PGEN ${BASE_PGENS})
    get_filename_component(PGEN_NAME ${PGEN} DIRECTORY)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/pgens/" "" PGEN_NAME
      ${PGEN_NAME})
    list(APPEND available_pgens ${PGEN_NAME})
  endforeach()

  # @TODO: for now, dropping support for Entity pgens
  # file(GLOB_RECURSE EXTRA_PGENS
  #   "${CMAKE_CURRENT_SOURCE_DIR}/extern/entity-pgens/**/pgen.hpp")
  # foreach(EXTRA_PGEN ${EXTRA_PGENS})
  #   get_filename_component(EXTRA_PGEN_NAME ${EXTRA_PGEN} DIRECTORY)
  #   string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/extern/entity-pgens/" ""
  #     EXTRA_PGEN_NAME ${EXTRA_PGEN_NAME})
  #   list(APPEND available_pgens "pgens/${EXTRA_PGEN_NAME}")
  # endforeach()

  set(available_pgens
    ${available_pgens}
    PARENT_SCOPE)
endfunction()

function(set_problem_generator pgen_name)
  if(pgen_name STREQUAL ".")
    message(FATAL_ERROR "Problem generator not specified")
  endif()

  list(FIND available_pgens ${pgen_name} PGEN_FOUND_IN_AVAILABLE)

  if(${PGEN_FOUND_IN_AVAILABLE} EQUAL -1)
    get_filename_component(pgen_path ${pgen_name} ABSOLUTE)
    string(REGEX REPLACE ".*/" "" pgen_name ${pgen_name})
    if(${single_pgen_mode})
      set(available_pgens ${pgen_name})
    else()
      list(APPEND available_pgens ${pgen_name})
    endif()
  else()
    set(pgen_path "${CMAKE_CURRENT_SOURCE_DIR}/pgens/${pgen_name}")
  endif()

  file(GLOB_RECURSE PGEN_FILE_FOUND "${pgen_path}/pgen.hpp")
  if(NOT PGEN_FILE_FOUND)
    message(FATAL_ERROR "pgen.hpp file not found in ${pgen_path}")
  endif()

  set(PGEN_TARGET ntt_pgen${pgen_suffix})
  add_library(${PGEN_TARGET} INTERFACE)
  target_link_libraries(${PGEN_TARGET} INTERFACE ntt_global ntt_framework
    ntt_archetypes ntt_kernels)

  target_include_directories(${PGEN_TARGET} INTERFACE ${pgen_path})

  set(PGEN
    ${pgen_name}
    PARENT_SCOPE)
  set(PGEN_FOUND
    TRUE
    PARENT_SCOPE)
  set(available_pgens
    ${available_pgens}
    PARENT_SCOPE)
endfunction()
