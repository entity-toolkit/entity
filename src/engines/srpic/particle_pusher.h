/**
 * @file engines/srpic/particle_pusher.h
 * @brief Particle pusher routines for the SRPIC engine
 * @implements
 *   - ntt::srpic::ParticlePush<> -> void
 * @namespaces:
 *   - ntt::srpic::
 */

#ifndef ENGINES_SRPIC_PARTICLE_PUSHER_H
#define ENGINES_SRPIC_PARTICLE_PUSHER_H

#include "enums.h"
#include "global.h"

#include "traits/metric.h"
#include "traits/policies.h"
#include "utils/log.h"
#include "utils/numeric.h"
#include "utils/param_container.h"

#include "engines/srpic/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/grid.h"
#include "framework/parameters/parameters.h"
#include "kernels/pushers/context.h"
#include "kernels/pushers/sr.hpp"
#include "kernels/pushers/sr_policies.h"

#include "engines/engine.hpp"

namespace ntt {
  namespace srpic {

    template <SRMetricClass M, PGenClass<SimEngine::SRPIC, M> PG>
    void ParticlePush(Domain<SimEngine::SRPIC, M>& domain,
                      const Grid<M::Dim>&          global_grid,
                      const M&                     global_metric,
                      const prm::Parameters&       engine_params,
                      const SimulationParams&      params,
                      const PG&                    pgen) {
      const auto dt   = engine_params.get<real_t>("dt");
      const auto time = engine_params.get<simtime_t>("time");

      real_t gx1 { ZERO }, gx2 { ZERO }, gx3 { ZERO }, ds { ZERO };
      real_t x_surf { ZERO };
      bool   has_atmosphere = false;
      for (auto& direction : dir::Directions<M::Dim>::orth) {
        if (global_grid.prtl_bc_in(direction) == PrtlBC::ATMOSPHERE) {
          raise::ErrorIf(has_atmosphere,
                         "Only one direction is allowed to have atm boundaries",
                         HERE);
          has_atmosphere = true;
          const auto g   = params.template get<real_t>(
            "grid.boundaries.atmosphere.g");
          ds = params.template get<real_t>("grid.boundaries.atmosphere.ds");
          const auto [sign, dim, xg_min, xg_max] =
            GetAtmosphereExtent(direction, global_metric, global_grid, params);
          if (dim == in::x1) {
            gx1 = sign > 0 ? g : -g;
            gx2 = ZERO;
            gx3 = ZERO;
          } else if (dim == in::x2) {
            gx1 = ZERO;
            gx2 = sign > 0 ? g : -g;
            gx3 = ZERO;
          } else if (dim == in::x3) {
            gx1 = ZERO;
            gx2 = ZERO;
            gx3 = sign > 0 ? g : -g;
          } else {
            raise::Error("Invalid dimension", HERE);
          }
          if (sign > 0) {
            x_surf = xg_min;
          } else {
            x_surf = xg_max;
          }
        }
      }
      for (auto& species : domain.species) {
        if ((species.pusher() == ParticlePusher::NONE) or (species.npart() == 0)) {
          continue;
        }
        species.set_unsorted();
        logger::Checkpoint(
          fmt::format("Launching particle pusher kernel for %d [%s] : %lu",
                      species.index(),
                      species.label().c_str(),
                      species.npart()),
          HERE);

        kernel::sr::PusherContext pusher_ctx {
          species.index(),
          species.pusher(),
          species.radiative_drag_flags(),
          species.mass(),
          species.charge(),
          time,
          dt,
          params.template get<real_t>("scales.omegaB0"),
          static_cast<int>(domain.mesh.n_active(in::x1)),
          static_cast<int>(domain.mesh.n_active(in::x2)),
          static_cast<int>(domain.mesh.n_active(in::x3))
        };

        if (species.pusher() & ParticlePusher::GCA) {
          pusher_ctx.gca = kernel::sr::PusherGCAContext(
            params.template get<real_t>("algorithms.gca.larmor_max"),
            params.template get<real_t>("algorithms.gca.e_ovr_b_max"));
        }

        if (has_atmosphere) {
          pusher_ctx.atmosphere = kernel::sr::PusherAtmosphereContext(gx1,
                                                                      gx2,
                                                                      gx3,
                                                                      x_surf,
                                                                      ds);
        }

        if (species.radiative_drag_flags() & RadiativeDrag::SYNCHROTRON) {
          pusher_ctx.synchrotron_drag = kernel::sr::PusherSynchrotronDragContext(
            dt,
            pusher_ctx.omegaB0,
            params.template get<real_t>("radiation.drag.synchrotron.gamma_rad"),
            species.mass());
        }

        if (species.radiative_drag_flags() & RadiativeDrag::COMPTON) {
          pusher_ctx.compton_drag = kernel::sr::PusherComptonDragContext(
            dt,
            pusher_ctx.omegaB0,
            params.template get<real_t>("radiation.drag.compton.gamma_rad"),
            species.mass());
        }

        auto pusher_boundaries = kernel::sr::PusherBoundaries<M::Dim> {
          domain.mesh.prtl_bc()
        };

        auto pusher_arrays = species.PusherKernelArrays();

        kernel::sr::MakePusherPolicy<decltype(domain.mesh.metric),
                                     decltype(domain),
                                     decltype(pgen)>(
          pgen,
          domain,
          params,
          pusher_ctx,
          species.emission_policy_flag(),
          has_atmosphere,
          [&](const auto& policies) {
            using policy_t = std::decay_t<decltype(policies)>;
            Kokkos::parallel_for(
              "ParticlePusher",
              species.rangeActiveParticles(),
              kernel::sr::Pusher_kernel<M, policy_t> { pusher_ctx,
                                                       pusher_boundaries,
                                                       pusher_arrays,
                                                       domain.fields.em,
                                                       domain.mesh.metric,
                                                       policies });
            // if emission takes place, update the npart and counter of emitted species
            if constexpr (
              not ::traits::emission::IsNoPolicy<typename policy_t::EmissionPolicy>) {
              const auto& emission_policy = policies.emission_policy;
              const auto emitted_species = emission_policy.emitted_species_indices();
              const auto n_inj = emission_policy.numbers_injected();
              raise::ErrorIf(emitted_species.size() != n_inj.size(),
                             "Emission policy emitted_species_indices and "
                             "numbers_injected must have the same size",
                             HERE);
              for (auto i = 0u; i < emitted_species.size(); ++i) {
                const auto sp_idx = emitted_species[i];
                raise::ErrorIf(sp_idx > domain.species.size(),
                               "Invalid emitted species index from custom "
                               "emission policy",
                               HERE);
                domain.species[sp_idx - 1].set_npart(
                  domain.species[sp_idx - 1].npart() + n_inj[i]);
                domain.species[sp_idx - 1].set_counter(
                  domain.species[sp_idx - 1].counter() + n_inj[i]);
              }
            }
          });
      }
    }

  } // namespace srpic
} // namespace ntt

#endif // ENGINES_SRPIC_PARTICLE_PUSHER_H
