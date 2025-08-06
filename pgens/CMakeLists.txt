# ------------------------------
# @defines: ntt_pgen [INTERFACE]
#
# @includes:
#
# * ../src/
# ------------------------------

add_library(ntt_pgen INTERFACE)
target_link_libraries(ntt_pgen INTERFACE ntt_global ntt_framework
                                         ntt_archetypes ntt_kernels)

target_include_directories(ntt_pgen
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/${PGEN})
