# ----------------------------- Simulation engine ---------------------------- #
function(set_engine engine_name)
  list(FIND simulation_engines ${engine_name} ENGINE_FOUND)

  if(${ENGINE_FOUND} EQUAL -1)
    message(FATAL_ERROR "Invalid simulation engine: ${engine_name}\nValid options are: ${simulation_engines}")
  else()
    set(ENGINE_FLAG ${engine}_engine)
    string(TOUPPER ${ENGINE_FLAG} ENGINE_FLAG)
    add_compile_options("-D ${ENGINE_FLAG}")
  endif()

  if(${engine_name} STREQUAL "sandbox")
    set(default_metric "minkowski" CACHE STRING "Default metric")
    set(metrics ${sr_metrics} ${gr_metrics} CACHE STRING "Metrics")
  elseif(${engine_name} STREQUAL "pic")
    set(default_metric "minkowski" CACHE STRING "Default metric")
    set(metrics ${sr_metrics} CACHE STRING "Metrics")
  elseif(${engine_name} STREQUAL "grpic")
    set(default_metric "qkerr_schild" CACHE STRING "Default metric")
    set(metrics ${gr_metrics} CACHE STRING "Metrics")
  endif()
endfunction()

# -------------------------------- Precision ------------------------------- #
function(set_precision precision_name)
  list(FIND precisions ${precision_name} PRECISION_FOUND)

  if(${PRECISION_FOUND} EQUAL -1)
    message(FATAL_ERROR "Invalid precision: ${precision_name}\nValid options are: ${precisions}")
  endif()

  if(${precision_name} STREQUAL "single")
    add_compile_options("-DSINGLE_PRECISION")
  endif()
endfunction()

# --------------------------------- Metric --------------------------------- #
function(set_metric metric_name)
  list(FIND metrics ${metric_name} METRIC_FOUND)

  if(${METRIC_FOUND} EQUAL -1)
    message(FATAL_ERROR "Invalid metric: ${metric_name}\nValid options are: ${metrics}")
  else()
    set(METRIC_FLAG ${metric}_metric)
    string(TOUPPER ${METRIC_FLAG} METRIC_FLAG)
    set(SIMULATION_METRIC ${metric})

    add_compile_options("-D ${METRIC_FLAG}")
    add_compile_options("-D SIMULATION_METRIC=\"${SIMULATION_METRIC}\"")
    add_compile_options("-D METRIC_HEADER=\"metrics/${metric}.h\"")
  endif()
endfunction()

# ---------------------------- Problem generator --------------------------- #
function(set_problem_generator pgen_name engine_name)
  if(NOT ${engine_name} STREQUAL "sandbox")
    set(PGEN_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/engines/${engine_name}/pgen/)
    file(GLOB_RECURSE PGENS ${PGEN_DIRECTORY}*.hpp)
    set(problem_generators "")

    foreach(pgen_file ${PGENS})
      string(REPLACE ${PGEN_DIRECTORY} "" new_pgen ${pgen_file})
      string(REPLACE ".hpp" "" new_pgen ${new_pgen})
      list(APPEND problem_generators ${new_pgen})
    endforeach()

    set(problem_generators ${problem_generators} CACHE STRING "Problem generators")

    list(FIND problem_generators ${pgen_name} PGEN_FOUND)

    if(${PGEN_FOUND} EQUAL -1)
      message(FATAL_ERROR "Problem generator ${pgen}.hpp not found\nAvailable problem generators: ${problem_generators}.")
    else()
      add_compile_options("-D PGEN_HEADER=\"${PGEN_DIRECTORY}/${pgen}.hpp\"")
      set(PGEN_FOUND TRUE CACHE BOOL "Problem generator found")
    endif()
  endif()
endfunction()

# ----------------------------- External force ---------------------------- #
function(set_ext_force ext_force_flag)
  if(${ext_force_flag})
    add_compile_options("-D EXTERNAL_FORCE")
  endif()
endfunction()