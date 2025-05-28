#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "framework/containers/particles.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <filesystem>

namespace ntt {

  template <SimEngine::type S, class M>
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
    const auto filename = params.template get<std::string>("simulation.name") +
                          "_stats.csv";
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
    g_stats_writer.init(
      params.template get<timestep_t>("output.stats.interval"),
      params.template get<simtime_t>("output.stats.interval_time"));
    g_stats_writer.defineStatsFilename(filename);
    g_stats_writer.defineStatsOutputs(stats_to_write);

    if (not std::filesystem::exists(filename)) {
      g_stats_writer.writeHeader();
    }
  }

  template <SimEngine::type S, class M, typename T, StatsID::type F>
  auto ComputeMoments(const SimulationParams& params,
                      const Mesh<M>&          mesh,
                      const std::vector<Particles<M::Dim, M::CoordType>>& prtl_species,
                      const std::vector<spidx_t>&        species,
                      const std::vector<unsigned short>& components) -> T {
    std::vector<spidx_t> specs = species;
    if (specs.size() == 0) {
      // if no species specified, take all massive species
      for (auto& sp : prtl_species) {
        if (sp.mass() > 0) {
          specs.push_back(sp.index());
        }
      }
    }
    for (const auto& sp : specs) {
      raise::ErrorIf((sp > prtl_species.size()) or (sp == 0),
                     "Invalid species index " + std::to_string(sp),
                     HERE);
    }
    // some parameters
    const auto use_weights = params.template get<bool>("particles.use_weights");
    const auto inv_n0      = ONE / params.template get<real_t>("scales.n0");

    T buffer = static_cast<T>(0);
    for (const auto& sp : specs) {
      auto& prtl_spec = prtl_species[sp - 1];
      // Kokkos::parallel_reduce(
      //   "ComputeMoments",
      //   prtl_spec.rangeActiveParticles(),
      //   // clang-format off
      //   kernel::ReducedParticleMoments_kernel<S, M, T, F>(components,
      //                                                     prtl_spec.i1, prtl_spec.i2, prtl_spec.i3,
      //                                                     prtl_spec.dx1, prtl_spec.dx2, prtl_spec.dx3,
      //                                                     prtl_spec.ux1, prtl_spec.ux2, prtl_spec.ux3,
      //                                                     prtl_spec.phi, prtl_spec.weight, prtl_spec.tag,
      //                                                     prtl_spec.mass(), prtl_spec.charge(),
      //                                                     use_weights, mesh.metric, mesh.flds_bc(), inv_n0),
      //   // clang-format on
      //   buffer);
    }
    return buffer;
  }

  template <SimEngine::type S, class M, StatsID::type F>
  auto ComputeFields(Domain<S, M>*                      domain,
                     const std::vector<unsigned short>& components) -> real_t {
    auto buffer { ZERO };
    // Kokkos::parallel_reduce(
    //   "ComputeMoments",
    //   prtl_spec.rangeActiveParticles(),
    //   kernel::ReducedFields_kernel<S, M, F>(components,
    //                                         domain->fields.em,
    //                                         domain->fields.cur,
    //                                         domain->mesh.metric),
    //   buffer);
    return buffer;
  }

  template <SimEngine::type S, class M>
  auto Metadomain<S, M>::WriteStats(const SimulationParams& params,
                                    timestep_t              current_step,
                                    timestep_t              finished_step,
                                    simtime_t               current_time,
                                    simtime_t finished_time) -> bool {
    if (not(params.template get<bool>("output.stats.enable") and
            g_stats_writer.shouldWrite(finished_step, finished_time))) {
      return false;
    }
    auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
    logger::Checkpoint("Writing stats", HERE);
    g_stats_writer.write(current_step);
    g_stats_writer.write(current_time);
    for (const auto& stat : g_stats_writer.statsWriters()) {
      if (stat.id() == StatsID::N) {
        g_stats_writer.write(
          ComputeMoments<S, M, real_t, StatsID::N>(params,
                                                   local_domain->mesh,
                                                   local_domain->species,
                                                   stat.species,
                                                   {}));
      } else if (stat.id() == StatsID::Npart) {
        g_stats_writer.write(
          ComputeMoments<S, M, npart_t, StatsID::Npart>(params,
                                                        local_domain->mesh,
                                                        local_domain->species,
                                                        stat.species,
                                                        {}));
      } else if (stat.id() == StatsID::Rho) {
        g_stats_writer.write(
          ComputeMoments<S, M, real_t, StatsID::Rho>(params,
                                                     local_domain->mesh,
                                                     local_domain->species,
                                                     stat.species,
                                                     {}));
      } else if (stat.id() == StatsID::Charge) {
        g_stats_writer.write(
          ComputeMoments<S, M, real_t, StatsID::Charge>(params,
                                                        local_domain->mesh,
                                                        local_domain->species,
                                                        stat.species,
                                                        {}));
      } else if (stat.id() == StatsID::T) {
        for (const auto& comp : stat.comp) {
          g_stats_writer.write(
            ComputeMoments<S, M, real_t, StatsID::T>(params,
                                                     local_domain->mesh,
                                                     local_domain->species,
                                                     stat.species,
                                                     comp));
        }
      } else if (stat.id() == StatsID::JdotE) {
        g_stats_writer.write(ComputeFields<S, M, StatsID::JdotE>(local_domain, {}));
      } else if (stat.id() == StatsID::E2) {
        g_stats_writer.write(ComputeFields<S, M, StatsID::E2>(local_domain, {}));
      } else if (stat.id() == StatsID::B2) {
        g_stats_writer.write(ComputeFields<S, M, StatsID::B2>(local_domain, {}));
      } else if (stat.id() == StatsID::ExB) {
        for (const auto& comp : stat.comp) {
          g_stats_writer.write(
            ComputeFields<S, M, StatsID::ExB>(local_domain, comp));
        }
      } else {
        raise::Error("Unrecognized stats ID " + stat.name(), HERE);
      }
    }
    g_stats_writer.endWriting();
    return true;
  }

  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_2D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_3D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Spherical<Dim::_2D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::QSpherical<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>>;

} // namespace ntt
