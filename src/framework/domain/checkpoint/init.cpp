#include "defaults.h"
#include "enums.h"
#include "global.h"

#include "traits/metric.h"
#include "utils/error.h"
#include "utils/formatting.h"

#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "framework/specialization_registry.h"
#include "output/checkpoint.h"

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <cstddef>
#include <string>
#include <vector>

namespace ntt {

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::InitCheckpointWriter(adios2::ADIOS*          ptr_adios,
                                              const SimulationParams& params) {
    raise::ErrorIf(ptr_adios == nullptr, "adios == nullptr", HERE);
    raise::ErrorIf(
      l_subdomain_indices().size() != 1,
      "Checkpoint writing for now is only supported for one subdomain per rank",
      HERE);
    auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
    raise::ErrorIf(local_domain->is_placeholder(),
                   "local_domain is a placeholder",
                   HERE);

    std::vector<ncells_t> glob_shape_with_ghosts, off_ncells_with_ghosts;
    for (auto d { 0u }; d < M::Dim; ++d) {
      off_ncells_with_ghosts.push_back(
        local_domain->offset_ncells()[d] +
        2 * N_GHOSTS * local_domain->offset_ndomains()[d]);
      glob_shape_with_ghosts.push_back(
        mesh().n_active()[d] + 2 * N_GHOSTS * ndomains_per_dim()[d]);
    }
    auto loc_shape_with_ghosts = local_domain->mesh.n_all();

    std::vector<unsigned short> npld_r, npld_i;
    for (auto s { 0u }; s < local_domain->species.size(); ++s) {
      npld_r.push_back(local_domain->species[s].npld_r());
      npld_i.push_back(local_domain->species[s].npld_i());
    }

    const path_t checkpoint_root = params.template get<std::string>(
      "checkpoint.write_path");

    g_checkpoint_writer.init(
      ptr_adios,
      checkpoint_root,
      params.template get<timestep_t>("checkpoint.interval"),
      params.template get<simtime_t>("checkpoint.interval_time"),
      params.template get<int>("checkpoint.keep"),
      params.template get<std::string>("checkpoint.walltime"),
      { params.template get<int>("adios2.aggregators_per_node",
                                 defaults::adios2::aggregators_per_node),
        params.template get<size_t>("adios2.max_shm_size",
                                    defaults::adios2::max_shm_size),
        params.template get<size_t>("adios2.buffer_chunk_size",
                                    defaults::adios2::buffer_chunk_size) });
    if (g_checkpoint_writer.enabled()) {
      local_domain->fields.CheckpointDeclare(g_checkpoint_writer.io(),
                                             loc_shape_with_ghosts,
                                             glob_shape_with_ghosts,
                                             off_ncells_with_ghosts);
      for (const auto& species : local_domain->species) {
        species.CheckpointDeclare(g_checkpoint_writer.io());
      }
      for (auto d { 0u }; d < M::Dim; ++d) {
        g_checkpoint_writer.io().DefineVariable<real_t>(
          fmt::format("subdomain_x%d_min", d + 1),
          { adios2::UnknownDim },
          { adios2::UnknownDim },
          { adios2::UnknownDim });
        g_checkpoint_writer.io().DefineVariable<real_t>(
          fmt::format("subdomain_x%d_max", d + 1),
          { adios2::UnknownDim },
          { adios2::UnknownDim },
          { adios2::UnknownDim });
        g_checkpoint_writer.io().DefineVariable<ncells_t>(
          fmt::format("subdomain_nx%d", d + 1),
          { adios2::UnknownDim },
          { adios2::UnknownDim },
          { adios2::UnknownDim });
      }
    }
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_CHECKPOINTS(S, M, D)                                        \
  template void Metadomain<S, M<D>>::InitCheckpointWriter(                     \
    adios2::ADIOS*,                                                            \
    const SimulationParams&);

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_CHECKPOINTS)
#undef METADOMAIN_CHECKPOINTS
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
