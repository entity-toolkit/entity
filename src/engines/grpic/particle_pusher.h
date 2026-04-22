#ifndef ENGINES_GRPIC_PARTICLE_PUSHER_H
#define ENGINES_GRPIC_PARTICLE_PUSHER_H

#include "enums.h"

#include "traits/metric.h"
#include "utils/log.h"
#include "utils/param_container.h"

#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"
#include "kernels/pushers/gr.hpp"

namespace ntt {
  namespace grpic {

    template <GRMetricClass M>
    void ParticlePush(Domain<SimEngine::GRPIC, M>& domain,
                      const SimulationParams&      params,
                      const prm::Parameters&       engine_params) {
      const auto dt = engine_params.get<real_t>("dt");
      for (auto& species : domain.species) {
        species.set_unsorted();
        logger::Checkpoint(
          fmt::format("Launching particle pusher kernel for %d [%s] : %lu",
                      species.index(),
                      species.label().c_str(),
                      species.npart()),
          HERE);
        if ((species.npart() == 0) or (species.pusher() == ParticlePusher::NONE)) {
          continue;
        }
        const auto pusher_ctx = kernel::gr::PusherContext {
          species.mass(),
          species.charge(),
          dt,
          params.template get<real_t>("scales.omegaB0"),
          params.template get<real_t>("algorithms.gr.pusher_eps"),
          params.template get<unsigned short>("algorithms.gr.pusher_niter"),
          static_cast<int>(domain.mesh.n_active(in::x1)),
          static_cast<int>(domain.mesh.n_active(in::x2)),
          static_cast<int>(domain.mesh.n_active(in::x3))
        };

        const auto pusher_boundaries = kernel::gr::PusherBoundaries<M::Dim> {
          domain.mesh.prtl_bc()
        };
        auto pusher_arrays = species.PusherKernelArrays();

        if (species.pusher() == ParticlePusher::PHOTON) {
          const auto range_policy =
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, kernel::gr::Massless_t>(
              0,
              species.npart());
          Kokkos::parallel_for("ParticlePusher",
                               range_policy,
                               kernel::gr::Pusher_kernel<M>(pusher_ctx,
                                                            pusher_boundaries,
                                                            pusher_arrays,
                                                            domain.fields.em,
                                                            domain.fields.em0,
                                                            domain.mesh.metric));
        } else if (species.pusher() == ParticlePusher::BORIS) {
          const auto range_policy =
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, kernel::gr::Massive_t>(
              0,
              species.npart());
          Kokkos::parallel_for("ParticlePusher",
                               range_policy,
                               kernel::gr::Pusher_kernel<M>(pusher_ctx,
                                                            pusher_boundaries,
                                                            pusher_arrays,
                                                            domain.fields.em,
                                                            domain.fields.em0,
                                                            domain.mesh.metric));
        } else {
          raise::Error("not implemented", HERE);
        }
      }
    }

  } // namespace grpic
} // namespace ntt

#endif // ENGINES_GRPIC_PARTICLE_PUSHER_H