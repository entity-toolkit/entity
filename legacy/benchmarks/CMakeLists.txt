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

# -------------------------------- Framework ------------------------------- #
file(GLOB_RECURSE FRAMEWORK_FILES ${SOURCE_DIR}/framework/*.cpp)
file(GLOB_RECURSE PIC_FILES ${SOURCE_DIR}/pic/*.cpp)
file(GLOB_RECURSE GRPIC_FILES ${SOURCE_DIR}/grpic/*.cpp)

# include framework headers for all targets
include_directories(${SOURCE_DIR}/framework)

set(all_metrics ${sr_metrics} ${gr_metrics})

# Libraries for all metrics and engines
# compile framework for all metrics and engines + all engines with corresponding metrics
foreach(metric ${all_metrics})
  list(FIND sr_metrics ${metric} sr_metric_index)

  if(NOT ${sr_metric_index} EQUAL -1)
    set(engine pic)
  else()
    set(engine grpic)
  endif()

  string(TOUPPER ${metric} metric_upper)
  string(TOUPPER ${engine} engine_upper)

  add_library(framework-${metric} STATIC ${FRAMEWORK_FILES})
  target_compile_options(framework-${metric} PUBLIC -D${metric_upper}_METRIC -D${engine_upper}_ENGINE -DSIMULATION_METRIC=\"${metric}\" -DMETRIC_HEADER=\"metrics/${metric}.h\")

  add_library(engine-${engine}-${metric} STATIC ${${engine_upper}_FILES})

  target_compile_options(engine-${engine}-${metric} PUBLIC -D${metric_upper}_METRIC -D${engine_upper}_ENGINE -DSIMULATION_METRIC=\"${metric}\" -DMETRIC_HEADER=\"metrics/${metric}.h\")
  target_compile_options(engine-${engine}-${metric} PUBLIC "-DPGEN_HEADER=\"pgen/dummy.hpp\"")
  target_include_directories(engine-${engine}-${metric} PRIVATE ${SOURCE_DIR}/${engine})
endforeach()

# ----------------------------------- Benchmarks ----------------------------------- #
function(define_benchmark metric engine title file)
  add_executable(bmark-${title}.xc ${file})
  target_link_libraries(bmark-${title}.xc PUBLIC framework-${metric} engine-${engine}-${metric})
  target_include_directories(bmark-${title}.xc PRIVATE ${SOURCE_DIR}/${engine})
endfunction()

define_benchmark("minkowski" "pic" "sr-mink" "sr-mink.cpp")
define_benchmark("spherical" "pic" "sr-sph" "sr-sph.cpp")
define_benchmark("qspherical" "pic" "sr-qsph" "sr-sph.cpp")
define_benchmark("kerr_schild" "grpic" "gr-ks" "gr.cpp")
define_benchmark("qkerr_schild" "grpic" "gr-qks" "gr.cpp")