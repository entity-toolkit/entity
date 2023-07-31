include(CTest)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

# include main source directory for all targets
include_directories(${SOURCE_DIR})

# --------------------------------- Wrapper -------------------------------- #
set(WRAPPER ${PROJECT_NAME}-wrapper)
add_library(${WRAPPER} STATIC ${SOURCE_DIR}/wrapper/kokkos.cpp)

# link wrapper with all targets
link_libraries(${WRAPPER})

# include wrapper header for all targets
include_directories(${SOURCE_DIR}/wrapper)

# -------------------------- Framework & Engines --------------------------- #

# include framework headers for all targets
include_directories(${SOURCE_DIR}/framework)

set(all_metrics ${sr_metrics} ${gr_metrics})

# Libraries for all metrics and engines
# compile framework for all metrics and engines + all engines with corresponding metrics
function(add_framework_library metric)
  string(TOUPPER ${metric} metric_upper)
  file(GLOB_RECURSE FRAMEWORK_FILES ${SOURCE_DIR}/framework/*.cpp)
  add_library(framework-${metric} STATIC EXCLUDE_FROM_ALL ${FRAMEWORK_FILES})
  target_compile_options(framework-${metric} PUBLIC -D${metric_upper}_METRIC -DSIMULATION_METRIC=\"${metric}\" -DMETRIC_HEADER=\"metrics/${metric}.h\")
endfunction()

function(add_engine_library metric engine)
  string(TOUPPER ${metric} metric_upper)
  string(TOUPPER ${engine} engine_upper)
  file(GLOB_RECURSE ${engine_upper}_FILES ${SOURCE_DIR}/engines/${engine}/*.cpp)
  add_library(engine-${engine}-${metric} STATIC EXCLUDE_FROM_ALL ${${engine_upper}_FILES})
  target_compile_options(engine-${engine}-${metric} PUBLIC -D${metric_upper}_METRIC -D${engine_upper}_ENGINE -DSIMULATION_METRIC=\"${metric}\" -DMETRIC_HEADER=\"metrics/${metric}.h\")
  target_compile_options(engine-${engine}-${metric} PUBLIC "-DPGEN_HEADER=\"pgen/dummy.hpp\"")
  target_include_directories(engine-${engine}-${metric} PRIVATE ${SOURCE_DIR}/engines ${SOURCE_DIR}/engines/${engine})
endfunction()

foreach(metric ${all_metrics})
  list(FIND sr_metrics ${metric} sr_metric_index)

  if(NOT ${sr_metric_index} EQUAL -1)
    set(engine pic)
  else()
    set(engine grpic)
  endif()

  string(TOUPPER ${metric} metric_upper)
  string(TOUPPER ${engine} engine_upper)

  add_framework_library(${metric})
  add_engine_library(${metric} ${engine})
  add_engine_library(${metric} sandbox)
endforeach()

# ---------------------------------- Tests --------------------------------- #
enable_testing()

# --------------------------------- Utils ---------------------------------- #
foreach(metric ${all_metrics})
  set(title utils-metadomain-${metric})
  add_executable(test-${title}.xc utils-metadomain.cpp)
  target_link_libraries(test-${title}.xc PUBLIC framework-${metric})
  add_test(NAME "Metadomain: ${metric}" COMMAND "test-${title}.xc")

  if(${output})
    set(title utils-writer-${metric})
    set(engine sandbox)
    add_executable(test-${title}.xc utils-writer.cpp)
    target_link_libraries(test-${title}.xc PUBLIC framework-${metric} engine-${engine}-${metric})
    target_include_directories(test-${title}.xc PRIVATE ${SOURCE_DIR}/engines/${engine})
    add_test(NAME "Writer: ${metric}" COMMAND "test-${title}.xc")
  endif()

  if(${mpi})
    set(title utils-comm-${metric})

    list(FIND sr_metrics ${metric} sr_metric_index)

    if(NOT ${sr_metric_index} EQUAL -1)
      set(engine pic)
    else()
      set(engine grpic)
    endif()

    add_executable(test-${title}.xc utils-comm.cpp)
    target_link_libraries(test-${title}.xc PUBLIC framework-${metric} engine-${engine}-${metric})
    target_include_directories(test-${title}.xc PRIVATE ${SOURCE_DIR}/engines/${engine})
    add_test(NAME "Comm: ${metric}" COMMAND "test-${title}.xc")
  endif()
endforeach()

# --------------------------------- Metrics -------------------------------- #
foreach(metric ${all_metrics})
  list(FIND sr_metrics ${metric} sr_metric_index)

  if(NOT ${sr_metric_index} EQUAL -1)
    if(${metric} STREQUAL "minkowski")
      set(filename_comp metric-comp-sr-mink.cpp)
    else()
      set(filename_comp metric-comp-sr-sph.cpp)
    endif()

  else()
    set(filename_comp metric-comp-gr.cpp)
  endif()

  set(title metric-trans-${metric})
  add_executable(test-${title}.xc metric-trans.cpp)
  target_link_libraries(test-${title}.xc PUBLIC framework-${metric})

  add_test(NAME "Vector/Coordinate Transformations: ${metric}" COMMAND "test-${title}.xc")

  set(title metric-comp-${metric})
  add_executable(test-${title}.xc ${filename_comp})
  target_link_libraries(test-${title}.xc PUBLIC framework-${metric})

  add_test(NAME "Metric Components: ${metric}" COMMAND "test-${title}.xc")
endforeach()

# ----------------------------------- SR ----------------------------------- #
set(metric minkowski)
set(engine pic)
set(title pusher-sr-minkowski)
add_executable(test-${title}.xc pusher-sr-mink.cpp)
target_link_libraries(test-${title}.xc PUBLIC framework-${metric} engine-${engine}-${metric})
target_include_directories(test-${title}.xc PRIVATE ${SOURCE_DIR}/engines/${engine})
add_test(NAME "Pusher: ${engine} ${metric}" COMMAND "test-${title}.xc")
