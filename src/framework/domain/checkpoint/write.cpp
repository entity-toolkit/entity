#include "enums.h"
#include "global.h"

#include "traits/metric.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "framework/specialization_registry.h"
#include "output/checkpoint.h"
#include "output/utils/writers.h"

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <cstddef>
#include <string>
#include <vector>

namespace ntt {

  template <SimEngine::type S, MetricClass M>
  auto Metadomain<S, M>::WriteCheckpoint(const SimulationParams& params,
                                         timestep_t              current_step,
                                         timestep_t              finished_step,
                                         simtime_t               current_time,
                                         simtime_t finished_time) -> bool {
    raise::ErrorIf(
      l_subdomain_indices().size() != 1,
      "Checkpointing for now is only supported for one subdomain per rank",
      HERE);
    if (not g_checkpoint_writer.shouldSave(finished_step, finished_time) or
        finished_step <= 1) {
      return false;
    }
    auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
    raise::ErrorIf(local_domain->is_placeholder(),
                   "local_domain is a placeholder",
                   HERE);
    logger::Checkpoint("Writing checkpoint", HERE);
    g_checkpoint_writer.beginSaving(current_step, current_time);
    {
      if (g_checkpoint_writer.written().empty()) {
        raise::Fatal("No checkpoint file to save metadata", HERE);
      }
      params.saveTOML(g_checkpoint_writer.written().back().second, current_time);

      // Recompute the local with-ghosts shape/offset every step so the
      // ADIOS variable selection tracks any rebalance that has happened
      // since InitCheckpointWriter.
      std::vector<ncells_t> loc_off_with_ghosts;
      for (auto d { 0u }; d < M::Dim; ++d) {
        loc_off_with_ghosts.push_back(
          local_domain->offset_ncells()[d] +
          2 * N_GHOSTS * local_domain->offset_ndomains()[d]);
      }
      local_domain->fields.CheckpointWrite(g_checkpoint_writer.io(),
                                           g_checkpoint_writer.writer(),
                                           local_domain->mesh.n_all(),
                                           loc_off_with_ghosts);
#if !defined(MPI_ENABLED)
      const std::size_t dom_tot = 1, dom_offset = 0;
#else
      const std::size_t dom_tot = g_mpi_size, dom_offset = g_mpi_rank;
#endif // MPI_ENABLED

      for (const auto& species : local_domain->species) {
        species.CheckpointWrite(g_checkpoint_writer.io(),
                                g_checkpoint_writer.writer(),
                                dom_tot,
                                dom_offset);
      }
      for (auto d { 0u }; d < M::Dim; ++d) {
        out::WriteVariable<real_t>(g_checkpoint_writer.io(),
                                   g_checkpoint_writer.writer(),
                                   fmt::format("subdomain_x%d_min", d + 1),
                                   local_domain->mesh.extent()[d].first,
                                   dom_tot,
                                   dom_offset);
        out::WriteVariable<real_t>(g_checkpoint_writer.io(),
                                   g_checkpoint_writer.writer(),
                                   fmt::format("subdomain_x%d_max", d + 1),
                                   local_domain->mesh.extent()[d].second,
                                   dom_tot,
                                   dom_offset);
        out::WriteVariable<ncells_t>(g_checkpoint_writer.io(),
                                     g_checkpoint_writer.writer(),
                                     fmt::format("subdomain_nx%d", d + 1),
                                     local_domain->mesh.n_active()[d],
                                     dom_tot,
                                     dom_offset);
      }
    }
    g_checkpoint_writer.endSaving();
    logger::Checkpoint("Checkpoint written", HERE);
    return true;
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_CHECKPOINTS(S, M, D)                                        \
  template auto Metadomain<S, M<D>>::WriteCheckpoint(const SimulationParams&,  \
                                                     timestep_t,               \
                                                     timestep_t,               \
                                                     simtime_t,                \
                                                     simtime_t) -> bool;

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_CHECKPOINTS)
#undef METADOMAIN_CHECKPOINTS
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
