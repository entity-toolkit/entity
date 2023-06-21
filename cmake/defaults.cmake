# ----------------------------- Defaults ---------------------------------- #
set(default_engine "pic" CACHE INTERNAL "Default engine")
set(default_precision "single" CACHE INTERNAL "Default precision")
set(default_pgen "dummy" CACHE INTERNAL "Default problem generator")
set(default_sr_metric "minkowski" CACHE INTERNAL "Default SR metric")
set(default_gr_metric "kerr_schild" CACHE INTERNAL "Default GR metric")

if(DEFINED ENV{Entity_ENABLE_OUTPUT})
  set(default_output $ENV{Entity_ENABLE_OUTPUT} CACHE INTERNAL "Default flag for output")
else()
  set(default_output OFF CACHE INTERNAL "Default flag for output")
endif()

set_property(CACHE default_output PROPERTY TYPE BOOL)

if(DEFINED ENV{Entity_ENABLE_GUI})
  set(default_gui $ENV{Entity_ENABLE_GUI} CACHE INTERNAL "Default flag for GUI")
else()
  set(default_gui OFF CACHE INTERNAL "Default flag for GUI")
endif()

set_property(CACHE default_gui PROPERTY TYPE BOOL)

if(DEFINED ENV{Kokkos_ENABLE_CUDA})
  set(default_KOKKOS_ENABLE_CUDA $ENV{Kokkos_ENABLE_CUDA} CACHE INTERNAL "Default flag for CUDA")
else()
  set(default_KOKKOS_ENABLE_CUDA OFF CACHE INTERNAL "Default flag for CUDA")
endif()

set_property(CACHE default_KOKKOS_ENABLE_CUDA PROPERTY TYPE BOOL)

if(DEFINED ENV{Kokkos_ENABLE_OPENMP})
  set(default_KOKKOS_ENABLE_OPENMP $ENV{Kokkos_ENABLE_OPENMP} CACHE INTERNAL "Default flag for OpenMP")
else()
  set(default_KOKKOS_ENABLE_OPENMP OFF CACHE INTERNAL "Default flag for OpenMP")
endif()

set_property(CACHE default_KOKKOS_ENABLE_OPENMP PROPERTY TYPE BOOL)
