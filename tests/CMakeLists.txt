include(CTest)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/)
include_directories(${SOURCE_DIR})

# --------------------------------- Wrapper -------------------------------- #
set(WRAPPER ${PROJECT_NAME}-wrapper)
add_library(${WRAPPER} STATIC ${SOURCE_DIR}/wrapper/kokkos.cpp)
link_libraries(${WRAPPER})
include_directories(${SOURCE_DIR}/wrapper)

# -------------------------------- Framework ------------------------------- #
file(GLOB_RECURSE FRAMEWORK_FILES ${SOURCE_DIR}/framework/*.cpp)
include_directories(${SOURCE_DIR}/framework)

# !TODO: maybe use shared LIBS?

# ---------------------------------- Tests --------------------------------- #
enable_testing()

function(configure_test test_target framework_target metric engine)
  string(TOUPPER ${metric} metric_upper)
  string(TOUPPER ${engine} engine_upper)
  add_library(${framework_target} STATIC ${FRAMEWORK_FILES})
  target_compile_options(${framework_target} PUBLIC -D${metric_upper}_METRIC -D${engine_upper}_ENGINE -DSIMULATION_METRIC=\"${metric}\" -DMETRIC_HEADER=\"metrics/${metric}.h\")
  target_link_libraries(${test_target} PUBLIC ${framework_target})
endfunction()

# --------------------------------- Metrics -------------------------------- #
set(all_metrics ${sr_metrics} ${gr_metrics})

set(sr_sph_metrics ${sr_metrics})
list(REMOVE_ITEM sr_sph_metrics minkowski)

foreach(metric ${all_metrics})
  list(FIND sr_metrics ${metric} sr_metric_index)

  if(NOT ${sr_metric_index} EQUAL -1)
    set(engine pic)

    if(${metric} STREQUAL "minkowski")
      set(filename_comp "metric-comp-sr-mink.cpp")
    else()
      set(filename_comp "metric-comp-sr-sph.cpp")
    endif()

  else()
    set(engine grpic)
    set(filename_comp "metric-comp-gr.cpp")
  endif()

  set(title "metric-trans-${metric}")
  add_executable(test-${title}.xc "metric-trans.cpp")
  configure_test(test-${title}.xc framework-${title} ${metric} ${engine})
  add_test(NAME "Vector/Coordinate Transformations: ${metric}" COMMAND "test-${title}.xc")

  set(title "metric-comp-${metric}")
  add_executable(test-${title}.xc ${filename_comp})
  configure_test(test-${title}.xc framework-${title} ${metric} ${engine})
  add_test(NAME "Metric Components: ${metric}" COMMAND "test-${title}.xc")
endforeach()