find_or_fetch_dependency(MPI FALSE REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
add_compile_options("-D MPI_ENABLED")
set(DEPENDENCIES ${DEPENDENCIES} MPI::MPI_CXX)
if(${DEVICE_ENABLED})
  if(${gpu_aware_mpi})
    add_compile_options("-D GPU_AWARE_MPI")

    # On Cray systems (e.g. Frontier) GPU-aware Cray MPICH can only handle
    # device pointers if the GPU Transport Layer (GTL) library is linked. The
    # Cray compiler wrappers (cc/CC) inject this automatically, but we build
    # with hipcc/nvcc directly, so find_package(MPI) only finds base libmpi and
    # the GTL is left out -> MPI_Sendrecv on a device pointer fails with "OFI
    # ... Bad address". Add it explicitly here.
    #
    # Cray PE exports PE_MPICH_GTL_DIR_<accel> / PE_MPICH_GTL_LIBS_<accel> (e.g.
    # amd_gfx90a -> -lmpi_gtl_hsa). Their absence means this is not a Cray MPICH
    # build, in which case nothing extra is needed.
    if("${Kokkos_DEVICES}" MATCHES "HIP")
      set(_gtl_accels amd_gfx942 amd_gfx940 amd_gfx90a amd_gfx908 amd_gfx906)
    elseif("${Kokkos_DEVICES}" MATCHES "CUDA")
      set(_gtl_accels nvidia90 nvidia80 nvidia70)
    elseif("${Kokkos_DEVICES}" MATCHES "SYCL")
      set(_gtl_accels ponteVecchio)
    else()
      set(_gtl_accels "")
    endif()

    set(_gtl_dir "")
    set(_gtl_libflag "")
    foreach(_accel ${_gtl_accels})
      if((NOT _gtl_dir) AND (DEFINED ENV{PE_MPICH_GTL_DIR_${_accel}}))
        # strip the leading "-L" from the Cray-provided value
        string(REGEX REPLACE "^-L" "" _gtl_dir
                             "$ENV{PE_MPICH_GTL_DIR_${_accel}}")
        string(REGEX REPLACE "^-l" "" _gtl_libflag
                             "$ENV{PE_MPICH_GTL_LIBS_${_accel}}")
      endif()
    endforeach()

    if(_gtl_dir AND _gtl_libflag)
      find_library(
        MPI_GTL_LIBRARY
        NAMES ${_gtl_libflag}
        HINTS "${_gtl_dir}"
        NO_DEFAULT_PATH)
      if(MPI_GTL_LIBRARY)
        message(
          STATUS "GPU-aware MPI: linking Cray GTL library ${MPI_GTL_LIBRARY}")
        set(DEPENDENCIES ${DEPENDENCIES} ${MPI_GTL_LIBRARY})
      else()
        message(
          FATAL_ERROR
            "${Red}gpu_aware_mpi=ON: Cray MPICH detected but the GTL "
            "library 'lib${_gtl_libflag}' was not found in '${_gtl_dir}'. "
            "GPU-aware MPI will crash at runtime without it. Make sure the "
            "craype-accel module is loaded, or build with gpu_aware_mpi=OFF."
            "${ColorReset}")
      endif()
    else()
      message(
        STATUS "GPU-aware MPI: no Cray GTL environment found; assuming the MPI "
               "implementation is GPU-aware without an extra transport library."
      )
    endif()
  endif()
else()
  set(gpu_aware_mpi
      OFF
      CACHE BOOL "Use explicit copy when using MPI + GPU")
endif()
