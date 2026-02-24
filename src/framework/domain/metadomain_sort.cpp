#include "enums.h"

#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "framework/specialization_registry.h"

#include <Kokkos_Core.hpp>

namespace ntt {

  template <SimEngine::type S, class M>
    requires IsCompatibleWithMetadomain<M>
  void Metadomain<S, M>::SortParticles(simtime_t,
                                       timestep_t              step,
                                       const SimulationParams& params,
                                       Domain<S, M>&           domain) const {
    const auto clear_interval = params.template get<timestep_t>(
      "particles.clear_interval");
    if ((clear_interval > 0u) and (step % clear_interval == 0u) and (step > 0u)) {
      for (auto& species : domain.species) {
        species.RemoveDead();
      }
    }
    for (auto& species : domain.species) {
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
