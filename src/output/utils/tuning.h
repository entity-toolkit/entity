/**
 * @file output/utils/tuning.h
 * @brief Functions for tuning ADIOS2 output, especially BP5 engine parameters
 * for large-scale parallel filesystems.
 * @implements
 *   - out::Bp5Tuning
 *   - out::Bp5ReadTuning
 *   - out::TotalAggregators -> int
 *   - out::ApplyBp5Tuning -> void
 *   - out::ApplyBp5ReadTuning -> void
 * @cpp:
 *   - tuning.cpp
 * @namespaces:
 *   - out::
 */
#ifndef OUTPUT_UTILS_TUNING_H
#define OUTPUT_UTILS_TUNING_H

#include "defaults.h"

#include <adios2.h>

#include <cstdlib>
#include <string>

namespace out {

  // BP5 tuning knobs sourced from the [adios2] toml section. Defaults mirror
  // ADIOS2's own built-ins; aggregators_per_node == 0 leaves the default of
  // one aggregator per node in place.
  struct Bp5Tuning {
    const int aggregators_per_node { ntt::defaults::adios2::aggregators_per_node };
    const size_t max_shm_size { ntt::defaults::adios2::max_shm_size };
    const size_t buffer_chunk_size { ntt::defaults::adios2::buffer_chunk_size };
  };

  // Total BP5 aggregator count for the current job = aggregators_per_node *
  // num_nodes (node count taken from MPI_COMM_TYPE_SHARED). Returns 0 when
  // aggregators_per_node <= 0, which leaves ADIOS2 on its built-in default.
  auto TotalAggregators(int aggregators_per_node) -> int;

  // Checkpoint-read knobs from the [adios2] toml section. The aggregation and
  // buffering parameters above are write-side only and are deliberately not
  // reused here: ADIOS2 accepts unknown parameters silently, so setting them
  // on a reader would look like tuning while doing nothing.
  struct Bp5ReadTuning {
    const int threads { ntt::defaults::adios2::read_threads };
    const int open_timeout_secs { ntt::defaults::adios2::read_open_timeout_secs };
    const int poll_secs { ntt::defaults::adios2::read_poll_secs };
  };

  // Apply the [adios2] BP5 tuning to a freshly declared IO whose engine is
  // BPFile/BP5. A no-op for other engines.
  void ApplyBp5Tuning(adios2::IO&, const std::string& engine, const Bp5Tuning&);

  // Apply the checkpoint-read knobs to a freshly declared reader IO whose
  // engine is BPFile/BP5. A no-op for other engines. `threads <= 0` leaves
  // ADIOS2 to size its own reader thread pool.
  void ApplyBp5ReadTuning(adios2::IO&,
                          const std::string& engine,
                          const Bp5ReadTuning&);

} // namespace out

#endif // OUTPUT_UTILS_TUNING_H
