# ----------------------------- Defaults ---------------------------------- #
set(default_engine "pic" CACHE INTERNAL "Default engine")
set(default_precision "single" CACHE INTERNAL "Default precision")
set(default_pgen "dummy" CACHE INTERNAL "Default problem generator")
set(default_sr_metric "minkowski" CACHE INTERNAL "Default SR metric")
set(default_gr_metric "kerr_schild" CACHE INTERNAL "Default GR metric")

set(default_output OFF CACHE INTERNAL "Default flag for output")
set_property(CACHE default_output PROPERTY TYPE BOOL)
set(default_nttiny OFF CACHE INTERNAL "Default flag for GUI")
set_property(CACHE default_nttiny PROPERTY TYPE BOOL)

set(default_KOKKOS_ENABLE_CUDA "OFF" CACHE INTERNAL BOOL "Default flag for CUDA")
set(default_KOKKOS_ENABLE_OPENMP "OFF" CACHE INTERNAL BOOL "Default flag for OpenMP")