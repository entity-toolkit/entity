/**
 * @file output/utils/tuning.h
 * @brief Functions for tuning ADIOS2 output, especially BP5 engine parameters
 * for large-scale parallel filesystems.
 * @implements
 *   - out::Bp5Tuning
 *   - out::TotalAggregators -> int
 *   - out::ApplyBp5Tuning -> void
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

  // Apply the [adios2] BP5 tuning to a freshly declared IO whose engine is
  // BPFile/BP5. A no-op for other engines.
  void ApplyBp5Tuning(adios2::IO&, const std::string& engine, const Bp5Tuning&);

} // namespace out

#endif // OUTPUT_UTILS_TUNING_H
