list(FIND tiled_deposit_tile_sizes "${tiled_deposit_tile_size}" _tps_idx)
if(_tps_idx EQUAL -1)
  message(
    FATAL_ERROR
      "${Red}tiled_deposit_tile_size must be one of ${tiled_deposit_tile_sizes}, "
      "got '${tiled_deposit_tile_size}'${ColorReset}")
endif()
add_compile_options("-D TILED_DEPOSIT")
add_compile_options("-D TILED_DEPOSIT_TILE_SIZE=${tiled_deposit_tile_size}")

# Compile-time tiled-deposit scratch halo drift. Sizes the halo so a particle
# that drifts up to DRIFT cells between two sorts still deposits inside its tile
# scratch; particles drifting further take the per-particle global-J escape
# valve (correct, only slower). This is independent of the sort cadence, which
# is set at runtime via `spatial_sorting_interval`. Defaults to 1 (the
# sorted-every-step case).
add_compile_options("-D TILED_DEPOSIT_DRIFT=${tiled_deposit_drift}")
