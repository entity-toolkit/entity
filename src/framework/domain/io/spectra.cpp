#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"

#include "framework/containers/particles.h"
#include "framework/domain/domain.h"
#include "framework/domain/mesh.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "framework/specialization_registry.h"
#include "kernels/particle_moments.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ntt {

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::WriteSpectra(const SimulationParams& params,
                                      Domain<S, M>*           local_domain,
                                      timestep_t              current_step,
                                      simtime_t               current_time) {

    g_writer.beginWriting(WriteMode::Spectra, current_step, current_time);
    const auto log_bins = params.template get<bool>("output.spectra.log_bins");
    const auto n_bins   = params.template get<size_t>("output.spectra.n_bins");
    auto       e_min    = params.template get<real_t>("output.spectra.e_min");
    auto       e_max    = params.template get<real_t>("output.spectra.e_max");
    if (log_bins) {
      e_min = math::log10(e_min);
      e_max = math::log10(e_max);
    }
    const array_t<real_t*> energy { "energy", n_bins + 1 };
    Kokkos::parallel_for(
      "GenerateEnergyBins",
      n_bins + 1,
      Lambda(uint32_t e) {
        if (log_bins) {
          energy(e) = math::pow(static_cast<real_t>(10),
                                e_min + (e_max - e_min) * static_cast<real_t>(e) /
                                          static_cast<real_t>(n_bins));
        } else {
          energy(e) = e_min + (e_max - e_min) * static_cast<real_t>(e) /
                                static_cast<real_t>(n_bins);
        }
      });
    for (const auto& spec : g_writer.spectraWriters()) {
      auto&            species = local_domain->species[spec.species() - 1];
      array_t<real_t*> dn { "dn", n_bins };
      auto dn_scatter = Kokkos::Experimental::create_scatter_view(dn);
      Kokkos::parallel_for(
        "ComputeSpectra",
        species.rangeActiveParticles(),
        kernel::ParticleDistribution_kernel<S, M> { species,
                                                    dn_scatter,
                                                    e_min,
                                                    e_max,
                                                    log_bins,
                                                    n_bins,
                                                    local_domain->mesh.metric });
      Kokkos::Experimental::contribute(dn, dn_scatter);
      g_writer.writeSpectrum(dn, spec.name());
    }
    g_writer.writeSpectrumBins(energy, "sEbn");
    g_writer.endWriting(WriteMode::Spectra);
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_OUTPUT(S, M, D)                                             \
  template void Metadomain<S, M<D>>::WriteSpectra(const SimulationParams&,     \
                                                  Domain<S, M<D>>*,            \
                                                  timestep_t,                  \
                                                  simtime_t);

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_OUTPUT)

#undef METADOMAIN_OUTPUT
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
