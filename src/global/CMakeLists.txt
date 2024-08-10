# ------------------------------
# @defines: ntt_global [STATIC/SHARED]
# @sources:
# - global.cpp
# - arch/kokkos_aliases.cpp
# - utils/cargs.cpp
# - utils/param_container.cpp
# @includes:
# - ./
# @uses:
# - kokkos [required]
# - plog [required]
# - mpi [optional]
# ------------------------------

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SOURCES 
  ${SRC_DIR}/global.cpp 
  ${SRC_DIR}/arch/kokkos_aliases.cpp 
  ${SRC_DIR}/utils/cargs.cpp
)
if (${output})
  list(APPEND SOURCES ${SRC_DIR}/utils/param_container.cpp)
endif()
add_library(ntt_global ${SOURCES})
target_include_directories(ntt_global
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(ntt_global PRIVATE stdc++fs)
