#include "enums.h"

#include "traits/metric.h"

#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "framework/specialization_registry.h"

#include <Kokkos_Core.hpp>

namespace ntt {

  template <SimEngine S, MetricClass M>
  void Metadomain<S, M>::SortParticles(simtime_t,
                                       timestep_t step,
                                       const SimulationParams&,
                                       Domain<S, M>& domain) const {
    for (auto& species : domain.species) {
      const auto clearing_interval = species.clearing_interval();
      if ((clearing_interval > 0u) and (step % clearing_interval == 0u) and
          (step > 0u)) {
        for (auto& species : domain.species) {
          species.RemoveDead();
        }
      }
      const auto spatial_sorting_interval = species.spatial_sorting_interval();
      if ((spatial_sorting_interval > 0u) and
          (step % spatial_sorting_interval == 0u)) {
        species.SortSpatially(domain.mesh);
      }
    }
  }

#define METADOMAIN_COMM(S, M, D)                                               \
  template void Metadomain<S, M<D>>::SortParticles(simtime_t,                  \
                                                   timestep_t,                 \
                                                   const SimulationParams&,    \
                                                   Domain<S, M<D>>&) const;

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_COMM)

#undef METADOMAIN_COMM

} // namespace ntt
