#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "checkpoint/reader.h"
#include "checkpoint/writer.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters.h"

namespace ntt {

  template <SimEngine::type S, class M>
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
      params.template get<std::string>("checkpoint.walltime"));
    if (g_checkpoint_writer.enabled()) {
      g_checkpoint_writer.defineFieldVariables(S,
                                               glob_shape_with_ghosts,
                                               off_ncells_with_ghosts,
                                               loc_shape_with_ghosts);
      for (auto& species : local_domain->species) {
        species.CheckpointDeclare(g_checkpoint_writer.io());
      }
    }
  }

  template <SimEngine::type S, class M>
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
      g_checkpoint_writer.saveAttrs(params, current_time);
      g_checkpoint_writer.saveField<M::Dim, 6>("em", local_domain->fields.em);
      if constexpr (S == SimEngine::GRPIC) {
        g_checkpoint_writer.saveField<M::Dim, 6>("em0", local_domain->fields.em0);
        g_checkpoint_writer.saveField<M::Dim, 3>("cur0", local_domain->fields.cur0);
      }
      std::size_t dom_tot = 1, dom_offset = 0;
#if defined(MPI_ENABLED)
      dom_tot    = g_mpi_size;
      dom_offset = g_mpi_rank;
#endif // MPI_ENABLED

      for (const auto& species : local_domain->species) {
        species.CheckpointWrite(g_checkpoint_writer.io(),
                                g_checkpoint_writer.writer(),
                                dom_tot,
                                dom_offset);
      }
    }
    g_checkpoint_writer.endSaving();
    logger::Checkpoint("Checkpoint written", HERE);
    return true;
  }

  template <SimEngine::type S, class M>
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
    for (const auto local_domain_idx : l_subdomain_indices()) {
      auto&                     domain = g_subdomains[local_domain_idx];
      adios2::Box<adios2::Dims> range;
      for (auto d { 0u }; d < M::Dim; ++d) {
        range.first.push_back(domain.offset_ncells()[d] +
                              2 * N_GHOSTS * domain.offset_ndomains()[d]);
        range.second.push_back(domain.mesh.n_all()[d]);
      }
      range.first.push_back(0);
      range.second.push_back(6);
      checkpoint::ReadFields<M::Dim, 6>(io, reader, "em", range, domain.fields.em);
      if constexpr (S == ntt::SimEngine::GRPIC) {
        checkpoint::ReadFields<M::Dim, 6>(io,
                                          reader,
                                          "em0",
                                          range,
                                          domain.fields.em0);
        adios2::Box<adios2::Dims> range3;
        for (auto d { 0u }; d < M::Dim; ++d) {
          range3.first.push_back(domain.offset_ncells()[d] +
                                 2 * N_GHOSTS * domain.offset_ndomains()[d]);
          range3.second.push_back(domain.mesh.n_all()[d]);
        }
        range3.first.push_back(0);
        range3.second.push_back(3);
        checkpoint::ReadFields<M::Dim, 3>(io,
                                          reader,
                                          "cur0",
                                          range3,
                                          domain.fields.cur0);
      }

      for (auto& species : domain.species) {
        species.CheckpointRead(io, reader, local_domain_idx, ndomains());
      }

    } // local subdomain loop

    reader.EndStep();
    reader.Close();
    logger::Checkpoint(
      fmt::format("Checkpoint reading done from %s", fname.c_str()),
      HERE);
  }

#define METADOMAIN_CHECKPOINTS(S, M)                                             \
  template void Metadomain<S, M>::InitCheckpointWriter(adios2::ADIOS*,           \
                                                       const SimulationParams&); \
  template auto Metadomain<S, M>::WriteCheckpoint(const SimulationParams&,       \
                                                  timestep_t,                    \
                                                  timestep_t,                    \
                                                  simtime_t,                     \
                                                  simtime_t) -> bool;            \
  template void Metadomain<S, M>::ContinueFromCheckpoint(adios2::ADIOS*,         \
                                                         const SimulationParams&);
  METADOMAIN_CHECKPOINTS(SimEngine::SRPIC, metric::Minkowski<Dim::_1D>)
  METADOMAIN_CHECKPOINTS(SimEngine::SRPIC, metric::Minkowski<Dim::_2D>)
  METADOMAIN_CHECKPOINTS(SimEngine::SRPIC, metric::Minkowski<Dim::_3D>)
  METADOMAIN_CHECKPOINTS(SimEngine::SRPIC, metric::Spherical<Dim::_2D>)
  METADOMAIN_CHECKPOINTS(SimEngine::SRPIC, metric::QSpherical<Dim::_2D>)
  METADOMAIN_CHECKPOINTS(SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>)
  METADOMAIN_CHECKPOINTS(SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>)
  METADOMAIN_CHECKPOINTS(SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>)
#undef METADOMAIN_CHECKPOINTS

} // namespace ntt
