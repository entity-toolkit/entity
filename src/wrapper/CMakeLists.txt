target_include_directories(${WRAPPER} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(${WRAPPER} PRIVATE kokkos.cpp)
target_link_libraries(${WRAPPER} PRIVATE Kokkos::kokkos)