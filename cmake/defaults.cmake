# cmake-lint: disable=C0103

# ----------------------------- Defaults ---------------------------------- #
if(DEFINED ENV{Entity_ENABLE_DEBUG})
  set(default_debug
      $ENV{Entity_ENABLE_DEBUG}
      CACHE INTERNAL "Default flag for debug mode")
else()
  set(default_debug
      OFF
      CACHE INTERNAL "Default flag for debug mode")
endif()

set_property(CACHE default_debug PROPERTY TYPE BOOL)

set(default_engine
    "pic"
    CACHE INTERNAL "Default engine")
set(default_precision
    "single"
    CACHE INTERNAL "Default precision")
set(default_pgen
    "."
    CACHE INTERNAL "Default problem generator")
set(default_sr_metric
    "minkowski"
    CACHE INTERNAL "Default SR metric")
set(default_gr_metric
    "kerr_schild"
    CACHE INTERNAL "Default GR metric")

if(DEFINED ENV{Entity_ENABLE_OUTPUT})
  set(default_output
      $ENV{Entity_ENABLE_OUTPUT}
      CACHE INTERNAL "Default flag for output")
else()
  set(default_output
      ON
      CACHE INTERNAL "Default flag for output")
endif()

set_property(CACHE default_output PROPERTY TYPE BOOL)

if(DEFINED ENV{Entity_ENABLE_GUI})
  set(default_gui
      $ENV{Entity_ENABLE_GUI}
      CACHE INTERNAL "Default flag for GUI")
else()
  set(default_gui
      OFF
      CACHE INTERNAL "Default flag for GUI")
endif()

set_property(CACHE default_gui PROPERTY TYPE BOOL)

if(DEFINED ENV{Entity_ENABLE_MPI})
  set(default_mpi
      $ENV{Entity_ENABLE_MPI}
      CACHE INTERNAL "Default flag for MPI")
else()
  set(default_mpi
      OFF
      CACHE INTERNAL "Default flag for MPI")
endif()

if(DEFINED ENV{Entity_MPI_DEVICE_COPY})
  set(default_mpi_device_copy
      $ENV{Entity_MPI_DEVICE_COPY}
      CACHE INTERNAL "Default flag for copying from device to host for MPI")
else()
  set(default_mpi_device_copy
      OFF
      CACHE INTERNAL "Default flag for copying from device to host for MPI")
endif()

set_property(CACHE default_mpi PROPERTY TYPE BOOL)

if(DEFINED ENV{Entity_ENABLE_GPU_AWARE_MPI})
  set(default_gpu_aware_mpi
      $ENV{Entity_ENABLE_GPU_AWARE_MPI}
      CACHE INTERNAL "Default flag for GPU-aware MPI")
else()
  set(default_gpu_aware_mpi
      ON
      CACHE INTERNAL "Default flag for GPU-aware MPI")
endif()

set_property(CACHE default_gpu_aware_mpi PROPERTY TYPE BOOL)
