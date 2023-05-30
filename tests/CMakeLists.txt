set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/)
include_directories(${SOURCE_DIR})

add_library(wrapper STATIC ${SOURCE_DIR}/wrapper/kokkos.cpp)
link_libraries(wrapper)
include_directories(${SOURCE_DIR}/wrapper)

file(GLOB_RECURSE FRAMEWORK_FILES ${SOURCE_DIR}/framework/*.cpp)
include_directories(${SOURCE_DIR}/framework)

function(configure_test target metric engine)
  string(TOUPPER ${metric} metric_upper)
  string(TOUPPER ${engine} engine_upper)
  target_compile_options(${target} PUBLIC -D${metric_upper}_METRIC -D${engine_upper}_ENGINE -DSIMULATION_METRIC=\"${metric}\" -DMETRIC_HEADER=\"metrics/${metric}.h\")
endfunction()

# GR KS test
add_executable(test-gr-ks.xc "gr-metric-ks.cpp")
add_library(framework-gr-ks STATIC ${FRAMEWORK_FILES})
configure_test(framework-gr-ks kerr_schild grpic)
target_link_libraries(test-gr-ks.xc PUBLIC framework-gr-ks)

# GR QKS test
add_executable(test-gr-qks.xc "gr-metric-qks.cpp")
add_library(framework-gr-qks STATIC ${FRAMEWORK_FILES})
configure_test(framework-gr-qks qkerr_schild grpic)
target_link_libraries(test-gr-qks.xc PUBLIC framework-gr-qks)

enable_testing()
add_test(NAME "GR Kerr_Schild" COMMAND "test-gr-ks.xc")
add_test(NAME "GR QKerr_Schild" COMMAND "test-gr-qks.xc")