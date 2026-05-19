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
#include "output/utils/readers.h"
#include "output/utils/writers.h"

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <algorithm>
#include <cstddef>
#include <limits>
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
      out::Bp5Tuning {
        params.template get<int>("adios2.aggregators_per_node"),
        params.template get<std::size_t>("adios2.max_shm_size"),
        params.template get<std::size_t>("adios2.buffer_chunk_size") });
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

      local_domain->fields.CheckpointWrite(g_checkpoint_writer.io(),
                                           g_checkpoint_writer.writer());
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

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::redecomposeFromCheckpoint(
    const std::vector<std::vector<ncells_t>>& dom_ncells,
    const std::vector<boundaries_t<real_t>>&  dom_extents) {

    // For each dimension: collect ncells per domain-grid position,
    // validate total is unchanged, then compute prefix-sum offsets.
    std::vector<std::vector<ncells_t>> offset_ncells_per_dom(
      g_ndomains,
      std::vector<ncells_t>(M::Dim, 0));

    for (auto d { 0u }; d < M::Dim; ++d) {
      std::vector<ncells_t> ncells_at_pos(g_ndomains_per_dim[d], 0);
      for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
        ncells_at_pos[g_domain_offsets[idx][d]] = dom_ncells[idx][d];
      }

      ncells_t total { 0 };
      for (const auto& n : ncells_at_pos) {
        total += n;
      }
      raise::ErrorIf(total != g_mesh.n_active()[d],
                     fmt::format("total cells in dim %d changed between "
                                 "checkpoint (%lu) and current (%lu); "
                                 "changing total domain size is not supported",
                                 d + 1,
                                 total,
                                 g_mesh.n_active()[d]),
                     HERE);

      ncells_t              running { 0 };
      std::vector<ncells_t> offset_at_pos(g_ndomains_per_dim[d], 0);
      for (unsigned int nd { 0 }; nd < g_ndomains_per_dim[d]; ++nd) {
        offset_at_pos[nd]  = running;
        running           += ncells_at_pos[nd];
      }
      for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
        offset_ncells_per_dom[idx][d] = offset_at_pos[g_domain_offsets[idx][d]];
      }
    }

    g_subdomains.clear();
    for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
      const auto& l_offset_ndomains = g_domain_offsets[idx];
      const auto& l_ncells          = dom_ncells[idx];
      const auto& l_offset_ncells   = offset_ncells_per_dom[idx];
      const auto& l_extent          = dom_extents[idx];

#if defined(MPI_ENABLED)
      const auto local = ((int)idx == g_mpi_rank);
      if (not local) {
        g_subdomains.emplace_back(false,
                                  idx,
                                  l_offset_ndomains,
                                  l_offset_ncells,
                                  l_ncells,
                                  l_extent,
                                  g_metric_params,
                                  g_species_params);
      } else {
        g_subdomains.emplace_back(idx,
                                  l_offset_ndomains,
                                  l_offset_ncells,
                                  l_ncells,
                                  l_extent,
                                  g_metric_params,
                                  g_species_params);
      }
      g_subdomains.back().set_mpi_rank(idx);
#else
      g_subdomains.emplace_back(idx,
                                l_offset_ndomains,
                                l_offset_ncells,
                                l_ncells,
                                l_extent,
                                g_metric_params,
                                g_species_params);
#endif
    }

    redefineNeighbors();
    redefineBoundaries();
  }

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::ContinueFromCheckpoint(adios2::ADIOS* ptr_adios,
                                                const SimulationParams& params) {
    raise::ErrorIf(ptr_adios == nullptr, "adios == nullptr", HERE);
    const path_t checkpoint_root = params.template get<std::string>(
      "checkpoint.read_path");
    const auto fname = checkpoint_root /
                       fmt::format("step-%08lu.bp",
                                   params.template get<timestep_t>(
                                     "checkpoint.start_step"));

    logger::Checkpoint(fmt::format("Reading checkpoint from %s", fname.c_str()),
                       HERE);

    adios2::IO io = ptr_adios->DeclareIO("Entity::CheckpointRead");
    io.SetEngine("BPFile");
#if !defined(MPI_ENABLED)
    adios2::Engine reader = io.Open(fname, adios2::Mode::Read);
#else
    adios2::Engine reader = io.Open(fname, adios2::Mode::Read, MPI_COMM_SELF);
#endif

    reader.BeginStep();

    // Phase 1: read all subdomain metadata to detect size changes
    std::vector<std::vector<ncells_t>> saved_ncells(g_ndomains,
                                                    std::vector<ncells_t>(M::Dim));
    std::vector<boundaries_t<real_t>> saved_extents(g_ndomains);
    boundaries_t<real_t>              global_extent;
    for (auto d { 0u }; d < M::Dim; ++d) {
      global_extent.emplace_back(std::numeric_limits<real_t>::max(),
                                 std::numeric_limits<real_t>::lowest());
    }

    bool needs_reconstruction = false;
    for (unsigned int dom_idx { 0 }; dom_idx < g_ndomains; ++dom_idx) {
      for (auto d { 0u }; d < M::Dim; ++d) {
        real_t x_min, x_max;
        out::ReadVariable<real_t>(io,
                                  reader,
                                  fmt::format("subdomain_x%d_min", d + 1),
                                  x_min,
                                  dom_idx);
        out::ReadVariable<real_t>(io,
                                  reader,
                                  fmt::format("subdomain_x%d_max", d + 1),
                                  x_max,
                                  dom_idx);
        saved_extents[dom_idx].emplace_back(x_min, x_max);
        global_extent[d].first  = std::min(global_extent[d].first, x_min);
        global_extent[d].second = std::max(global_extent[d].second, x_max);

        ncells_t nx;
        out::ReadVariable<ncells_t>(io,
                                    reader,
                                    fmt::format("subdomain_nx%d", d + 1),
                                    nx,
                                    dom_idx);
        saved_ncells[dom_idx][d] = nx;

        if (nx != subdomain_ptr(dom_idx)->mesh.n_active()[d]) {
          needs_reconstruction = true;
        }
      }
    }

    // Phase 2: update domain structure
    if (needs_reconstruction) {
      redecomposeFromCheckpoint(saved_ncells, saved_extents);
    } else {
      for (unsigned int dom_idx { 0 }; dom_idx < g_ndomains; ++dom_idx) {
        subdomain_ptr(dom_idx)->mesh.set_extent(saved_extents[dom_idx]);
      }
    }
    g_mesh.set_extent(global_extent);

    // Phase 3: read field and particle data using the (now-correct) domain layout
    for (const auto local_domain_idx : l_subdomain_indices()) {
      auto local_domain = subdomain_ptr(local_domain_idx);

      adios2::Box<adios2::Dims> range;
      for (auto d { 0u }; d < M::Dim; ++d) {
        range.first.push_back(local_domain->offset_ncells()[d] +
                              2 * N_GHOSTS * local_domain->offset_ndomains()[d]);
        range.second.push_back(local_domain->mesh.n_all()[d]);
      }
      local_domain->fields.CheckpointRead(io, reader, range);

      for (auto& species : local_domain->species) {
        species.CheckpointRead(io, reader, ndomains(), local_domain_idx);
      }
    }

    reader.EndStep();
    reader.Close();

    logger::Checkpoint(
      fmt::format("Checkpoint reading done from %s", fname.c_str()),
      HERE);
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_CHECKPOINTS(S, M, D)                                        \
  template void Metadomain<S, M<D>>::InitCheckpointWriter(                     \
    adios2::ADIOS*,                                                            \
    const SimulationParams&);                                                  \
  template auto Metadomain<S, M<D>>::WriteCheckpoint(const SimulationParams&,  \
                                                     timestep_t,               \
                                                     timestep_t,               \
                                                     simtime_t,                \
                                                     simtime_t) -> bool;       \
  template void Metadomain<S, M<D>>::ContinueFromCheckpoint(                   \
    adios2::ADIOS*,                                                            \
    const SimulationParams&);                                                  \
  template void Metadomain<S, M<D>>::redecomposeFromCheckpoint(                \
    const std::vector<std::vector<ncells_t>>&,                                 \
    const std::vector<boundaries_t<real_t>>&);
  NTT_FOREACH_SPECIALIZATION(METADOMAIN_CHECKPOINTS)
#undef METADOMAIN_CHECKPOINTS
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
