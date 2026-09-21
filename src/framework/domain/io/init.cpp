#include "defaults.h"
#include "enums.h"
#include "global.h"

#include "traits/metric.h"
#include "utils/error.h"

#include "framework/domain/domain.h"
#include "framework/domain/mesh.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "framework/specialization_registry.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <vector>

namespace ntt {

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::InitWriter(adios2::ADIOS*          ptr_adios,
                                    const SimulationParams& params) {
    raise::ErrorIf(
      l_subdomain_indices().size() != 1,
      "Output for now is only supported for one subdomain per rank",
      HERE);
    auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
    raise::ErrorIf(local_domain->is_placeholder(),
                   "local_domain is a placeholder",
                   HERE);
    const auto incl_ghosts = params.template get<bool>("output.debug.ghosts");

    auto glob_shape_with_ghosts = mesh().n_active();
    auto off_ncells_with_ghosts = local_domain->offset_ncells();
    auto off_ndomains           = local_domain->offset_ndomains();
    auto loc_shape_with_ghosts  = local_domain->mesh.n_active();
    if (incl_ghosts) {
      for (auto d { 0 }; d <= M::Dim; ++d) {
        glob_shape_with_ghosts[d] += 2 * N_GHOSTS * ndomains_per_dim()[d];
        off_ncells_with_ghosts[d] += 2 * N_GHOSTS * off_ndomains[d];
        loc_shape_with_ghosts[d]  += 2 * N_GHOSTS;
      }
    }

    g_writer.init(
      ptr_adios,
      params.template get<std::string>("output.format"),
      params.template get<std::string>("simulation.name"),
      { params.template get<int>("adios2.aggregators_per_node",
                                 defaults::adios2::aggregators_per_node),
        params.template get<size_t>("adios2.max_shm_size",
                                    defaults::adios2::max_shm_size),
        params.template get<size_t>("adios2.buffer_chunk_size",
                                    defaults::adios2::buffer_chunk_size) });
    g_writer.defineMeshLayout(glob_shape_with_ghosts,
                              off_ncells_with_ghosts,
                              loc_shape_with_ghosts,
                              { local_domain->index(), ndomains() },
                              params.template get<std::vector<unsigned int>>(
                                "output.fields.downsampling"),
                              incl_ghosts,
                              M::CoordType);
    const auto fields_to_write = params.template get<std::vector<std::string>>(
      "output.fields.quantities");
    const auto custom_fields_to_write = params.template get<std::vector<std::string>>(
      "output.fields.custom");
    std::vector<std::string> all_fields_to_write;
    std::merge(fields_to_write.begin(),
               fields_to_write.end(),
               custom_fields_to_write.begin(),
               custom_fields_to_write.end(),
               std::back_inserter(all_fields_to_write));
    const auto species_to_write = params.template get<std::vector<spidx_t>>(
      "output.particles.species");
    g_writer.defineFieldOutputs(S, all_fields_to_write);

    g_writer.clearSpeciesIndex();
    for (const auto& s : species_to_write) {
      g_writer.addSpeciesIndex(s);
    }
    for (const auto sp : g_writer.speciesIndices()) {
      local_domain->species[sp - 1].OutputDeclare(g_writer.io());
    }

    // spectra write all particle species
    std::vector<spidx_t> spectra_species {};
    for (const auto& sp : species_params()) {
      spectra_species.push_back(sp.index());
    }
    const auto num_spatial_bins = params.template get<std::vector<size_t>>(
      "output.spectra.num_spatial_bins");
    g_writer.defineSpectraOutputs(spectra_species, num_spatial_bins);
    for (const auto& type : { "fields", "particles", "spectra" }) {
      g_writer.addTracker(type,
                          params.template get<timestep_t>(
                            "output." + std::string(type) + ".interval"),
                          params.template get<simtime_t>(
                            "output." + std::string(type) + ".interval_time"));
    }
    g_writer.writeAttrs(params);
  }

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::InitStatsWriter(const SimulationParams& params,
                                         bool                    is_resuming) {
    raise::ErrorIf(
      l_subdomain_indices().size() != 1,
      "StatsWriter for now is only supported for one subdomain per rank",
      HERE);
    auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
    raise::ErrorIf(local_domain->is_placeholder(),
                   "local_domain is a placeholder",
                   HERE);
    const auto simname  = params.template get<std::string>("simulation.name");
    const auto filename = std::filesystem::path(simname) /
                          (simname + "_stats.csv");
    const auto enable_stats = params.template get<bool>("output.stats.enable");
    if (enable_stats and (not is_resuming)) {
      CallOnce(
        [](auto& filename) {
          if (std::filesystem::exists(filename)) {
            std::filesystem::remove(filename);
          }
        },
        filename);
    }
    const auto stats_to_write = params.template get<std::vector<std::string>>(
      "output.stats.quantities");
    const auto custom_stats_to_write = params.template get<std::vector<std::string>>(
      "output.stats.custom");
    g_stats_writer.init(
      params.template get<timestep_t>("output.stats.interval"),
      params.template get<simtime_t>("output.stats.interval_time"));
    g_stats_writer.defineStatsFilename(filename);
    g_stats_writer.defineStatsOutputs(stats_to_write, false);
    g_stats_writer.defineStatsOutputs(custom_stats_to_write, true);

    if (not std::filesystem::exists(filename)) {
      g_stats_writer.writeHeader();
    }
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_OUTPUT(S, M, D)                                             \
  template void Metadomain<S, M<D>>::InitWriter(adios2::ADIOS*,                \
                                                const SimulationParams&);      \
  template void Metadomain<S, M<D>>::InitStatsWriter(const SimulationParams&,  \
                                                     bool);

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_OUTPUT)

#undef METADOMAIN_OUTPUT
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
