# Vendor sort: oneDPL on SYCL, Thrust on CUDA, rocThrust/rocprim on HIP. When
# `vendor_sort` is ON (default) the available library is detected and used; the
# spatial sort then builds a single permutation that gathers all SoA members.
# When `vendor_sort` is OFF, or no library is found, the code falls back to
# Kokkos::BinSort, which sorts each member in place -- lower peak memory and no
# maxnpart gather buffer, at the cost of sort speed (negligible when sorting is
# a small fraction of the step). The `vendor_sort` knob lets you force the
# BinSort fallback even when a vendor library is present.
if("${Kokkos_DEVICES}" MATCHES "SYCL")
  find_package(oneDPL QUIET)
  if(oneDPL_FOUND)
    message(STATUS "oneDPL found, enabling SYCL sort_by_key")
    add_compile_options("-D ONEDPL_ENABLED")
    set(DEPENDENCIES ${DEPENDENCIES} oneDPL)
  else()
    message(STATUS "oneDPL not found; using BinSort fallback "
                   "for SYCL sort_by_key")
  endif()
elseif("${Kokkos_DEVICES}" MATCHES "CUDA")
  find_package(Thrust QUIET)
  if(Thrust_FOUND)
    message(STATUS "Thrust enabled for CUDA sort_by_key")
    add_compile_options("-D THRUST_ENABLED")
  else()
    message(STATUS "Thrust not found; using BinSort fallback "
                   "for CUDA sort_by_key")
  endif()
elseif("${Kokkos_DEVICES}" MATCHES "HIP")
  # rocThrust ships with ROCm. The HIP sort_by_key path uses rocprim's
  # bounded-bit radix sort directly (rocprim is rocThrust's own dependency, so
  # its headers come in transitively; we find it explicitly to keep the include
  # path robust). This builds a single permutation that gathers all SoA members,
  # instead of the legacy per-member Kokkos::BinSort path which allocates a
  # fresh `sorted_values` buffer for every member every step (the dominant
  # source of allocator churn / fragmentation on ROCm).
  find_package(rocthrust QUIET)
  if(rocthrust_FOUND)
    message(STATUS "rocThrust enabled for HIP sort_by_key")
    add_compile_options("-D ROCTHRUST_ENABLED")
    set(DEPENDENCIES ${DEPENDENCIES} roc::rocthrust)
    find_package(rocprim QUIET)
    if(rocprim_FOUND)
      set(DEPENDENCIES ${DEPENDENCIES} roc::rocprim)
    endif()
  else()
    message(STATUS "rocThrust not found; using BinSort "
                   "fallback for HIP sort_by_key")
  endif()
else()
  message(FATAL_ERROR "vendor_sort enabled, but device not recognized")
endif()
