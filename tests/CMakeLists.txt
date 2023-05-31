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

function(configure_test target metric engine)
  string(TOUPPER ${metric} metric_upper)
  string(TOUPPER ${engine} engine_upper)
  target_compile_options(${target} PUBLIC -D${metric_upper}_METRIC -D${engine_upper}_ENGINE -DSIMULATION_METRIC=\"${metric}\" -DMETRIC_HEADER=\"metrics/${metric}.h\")
endfunction()

# --------------------------------- Metrics -------------------------------- #
set(all_metrics ${sr_metrics} ${gr_metrics})
set(sph_metrics ${all_metrics})
list(REMOVE_ITEM sph_metrics minkowski)

foreach(metric ${all_metrics})
  list(FIND sr_metrics ${metric} sr_metric_index)

  if(NOT ${sr_metric_index} EQUAL -1)
    set(engine pic)
  else()
    set(engine grpic)
  endif()

  set(title "metric-all-${metric}")

  add_executable(test-${title}.xc "metric-all.cpp")
  add_library(framework-${title} STATIC ${FRAMEWORK_FILES})
  configure_test(framework-${title} ${metric} ${engine})
  target_link_libraries(test-${title}.xc PUBLIC framework-${title})
  add_test(NAME "Cov2Hat -> Hat2Cntrv -> Cntrv2Cov: ${metric}" COMMAND "test-${title}.xc")

  list(FIND sph_metrics ${metric} sph_metric_index)

  if(NOT ${sph_metric_index} EQUAL -1)
    set(title "metric-sph-${metric}")

    add_executable(test-${title}.xc "metric-sph.cpp")
    add_library(framework-${title} STATIC ${FRAMEWORK_FILES})
    configure_test(framework-${title} ${metric} ${engine})
    target_link_libraries(test-${title}.xc PUBLIC framework-${title})
    add_test(NAME "Code2Cart -> Cart2Code + Code2Sph -> Sph2Code: ${metric}" COMMAND "test-${title}.xc")
  endif()
endforeach()

# ------------------------------- GR KS test ------------------------------- #
add_executable(test-gr-ks.xc "gr-metric-ks.cpp")
add_library(framework-gr-ks STATIC ${FRAMEWORK_FILES})
configure_test(framework-gr-ks kerr_schild grpic)
target_link_libraries(test-gr-ks.xc PUBLIC framework-gr-ks)
add_test(NAME "GR Kerr_Schild" COMMAND "test-gr-ks.xc")

# ------------------------------- GR QKS test ------------------------------ #
add_executable(test-gr-qks.xc "gr-metric-qks.cpp")
add_library(framework-gr-qks STATIC ${FRAMEWORK_FILES})
configure_test(framework-gr-qks qkerr_schild grpic)
target_link_libraries(test-gr-qks.xc PUBLIC framework-gr-qks)
add_test(NAME "GR QKerr_Schild" COMMAND "test-gr-qks.xc")