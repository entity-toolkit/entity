list(FIND team_policy_tile_sizes "${team_policy_tile_size}" _tps_idx)
if(_tps_idx EQUAL -1)
  message(
    FATAL_ERROR
      "${Red}team_policy_tile_size must be one of ${team_policy_tile_sizes}, "
      "got '${team_policy_tile_size}'${ColorReset}")
endif()
add_compile_options("-D TEAM_POLICY")
add_compile_options("-D TEAM_POLICY_TILE_SIZE=${team_policy_tile_size}")

# Compile-time tiled-deposit scratch halo drift. Sizes the halo so a particle
# that drifts up to DRIFT cells between two sorts still deposits inside its tile
# scratch; particles drifting further take the per-particle global-J escape
# valve (correct, only slower). This is independent of the sort cadence, which
# is set at runtime via `spatial_sorting_interval`. Defaults to 1 (the
# sorted-every-step case).
add_compile_options("-D TEAM_POLICY_DRIFT=${team_policy_drift}")

# Vendor sort: oneDPL on SYCL, Thrust on CUDA, rocThrust/rocprim on HIP. When
# `vendor_sort` is ON (default) the available library is detected and used; the
# spatial sort then builds a single permutation that gathers all SoA members.
# When `vendor_sort` is OFF, or no library is found, the code falls back to
# Kokkos::BinSort, which sorts each member in place -- lower peak memory and no
# maxnpart gather buffer, at the cost of sort speed (negligible when sorting is
# a small fraction of the step). The `vendor_sort` knob lets you force the
# BinSort fallback even when a vendor library is present.
if(${vendor_sort})
  if("${Kokkos_DEVICES}" MATCHES "SYCL")
    find_package(oneDPL QUIET)
    if(oneDPL_FOUND)
      message(STATUS "team_policy: oneDPL found, enabling SYCL sort_by_key")
      add_compile_options("-D ONEDPL_ENABLED")
      set(DEPENDENCIES ${DEPENDENCIES} oneDPL)
    else()
      message(STATUS "team_policy: oneDPL not found; using BinSort fallback "
                     "for SYCL sort_by_key")
    endif()
  endif()

  if("${Kokkos_DEVICES}" MATCHES "CUDA")
    find_package(Thrust QUIET)
    if(Thrust_FOUND)
      message(STATUS "team_policy: Thrust enabled for CUDA sort_by_key")
      add_compile_options("-D THRUST_ENABLED")
    else()
      message(STATUS "team_policy: Thrust not found; using BinSort fallback "
                     "for CUDA sort_by_key")
    endif()
  endif()

  if("${Kokkos_DEVICES}" MATCHES "HIP")
    # rocThrust ships with ROCm. The HIP sort_by_key path uses rocprim's
    # bounded-bit radix sort directly (rocprim is rocThrust's own dependency, so
    # its headers come in transitively; we find it explicitly to keep the
    # include path robust). This builds a single permutation that gathers all
    # SoA members, instead of the legacy per-member Kokkos::BinSort path which
    # allocates a fresh `sorted_values` buffer for every member every step (the
    # dominant source of allocator churn / fragmentation on ROCm).
    find_package(rocthrust QUIET)
    if(rocthrust_FOUND)
      message(STATUS "team_policy: rocThrust enabled for HIP sort_by_key")
      add_compile_options("-D ROCTHRUST_ENABLED")
      set(DEPENDENCIES ${DEPENDENCIES} roc::rocthrust)
      find_package(rocprim QUIET)
      if(rocprim_FOUND)
        set(DEPENDENCIES ${DEPENDENCIES} roc::rocprim)
      endif()
    else()
      message(STATUS "team_policy: rocThrust not found; using BinSort "
                     "fallback for HIP sort_by_key")
    endif()
  endif()
else()
  message(STATUS "team_policy: vendor_sort=OFF; forcing Kokkos::BinSort "
                 "fallback for spatial sort_by_key")
endif()
