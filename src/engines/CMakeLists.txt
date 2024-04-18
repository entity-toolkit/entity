# ------------------------------
# @defines: ntt_engines [STATIC/SHARED]
# @sources:
# - engine_printer.cpp
# - engine_init.cpp
# - srpic/srpic.cpp
# - grpic/grpic.cpp
# @includes:
# - ../
# @depends:
# - ntt_global [required]
# - ntt_framework [required]
# - ntt_metrics [required]
# - ntt_kernels [required]
# - ntt_pgen [required]
# @uses:
# - kokkos [required]
# - plog [required]
# - adios2 [optional]
# - mpi [optional]
# ------------------------------

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SOURCES 
  ${SRC_DIR}/engine_printer.cpp
  ${SRC_DIR}/engine_init.cpp
  ${SRC_DIR}/srpic/srpic.cpp
  ${SRC_DIR}/grpic/grpic.cpp
)
add_library(ntt_engines ${SOURCES})

set(libs ntt_global ntt_framework ntt_metrics ntt_kernels ntt_pgen)
add_dependencies(ntt_engines ${libs})
target_link_libraries(ntt_engines PUBLIC ${libs})
target_compile_definitions(ntt_engines PRIVATE PGEN=\"${PGEN}\")

target_include_directories(ntt_engines
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../
)