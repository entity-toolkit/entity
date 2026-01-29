#ifndef ENGINES_SRPIC_PARTICLE_PUSHER_H
#define ENGINES_SRPIC_PARTICLE_PUSHER_H

#include "enums.h"
#include "global.h"

#include "utils/log.h"
#include "utils/numeric.h"
#include "utils/param_container.h"

#include "metrics/traits.h"

#include "archetypes/traits.h"
#include "engines/srpic/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/grid.h"
#include "framework/parameters/parameters.h"

#include "kernels/particle_pusher_sr.hpp"

namespace ntt {
  namespace srpic {

    template <class M, class PG>
      requires metric::traits::HasD<M>
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

        kernel::sr::PusherParams pusher_params {};
        pusher_params.pusher_flags         = species.pusher();
        pusher_params.radiative_drag_flags = species.radiative_drag_flags();
        pusher_params.mass                 = species.mass();
        pusher_params.charge               = species.charge();
        pusher_params.time                 = time;
        pusher_params.dt                   = dt;
        pusher_params.omegaB0 = params.template get<real_t>("scales.omegaB0");
        pusher_params.ni1     = domain.mesh.n_active(in::x1);
        pusher_params.ni2     = domain.mesh.n_active(in::x2);
        pusher_params.ni3     = domain.mesh.n_active(in::x3);
        pusher_params.boundaries = domain.mesh.prtl_bc();

        if (species.pusher() & ParticlePusher::GCA) {
          pusher_params.gca_params.set(
            "larmor_max",
            params.template get<real_t>("algorithms.gca.larmor_max"));
          pusher_params.gca_params.set(
            "e_ovr_b_max",
            params.template get<real_t>("algorithms.gca.e_ovr_b_max"));
        }

        if (species.radiative_drag_flags() & RadiativeDrag::SYNCHROTRON) {
          pusher_params.radiative_drag_params.set(
            "synchrotron_gamma_rad",
            params.template get<real_t>(
              "radiation.drag.synchrotron.gamma_rad"));
        }

        if (species.radiative_drag_flags() & RadiativeDrag::COMPTON) {
          pusher_params.radiative_drag_params.set(
            "compton_gamma_rad",
            params.template get<real_t>("radiation.drag.compton.gamma_rad"));
        }

        kernel::sr::PusherArrays pusher_arrays {};
        pusher_arrays.sp       = species.index();
        pusher_arrays.i1       = species.i1;
        pusher_arrays.i2       = species.i2;
        pusher_arrays.i3       = species.i3;
        pusher_arrays.i1_prev  = species.i1_prev;
        pusher_arrays.i2_prev  = species.i2_prev;
        pusher_arrays.i3_prev  = species.i3_prev;
        pusher_arrays.dx1      = species.dx1;
        pusher_arrays.dx2      = species.dx2;
        pusher_arrays.dx3      = species.dx3;
        pusher_arrays.dx1_prev = species.dx1_prev;
        pusher_arrays.dx2_prev = species.dx2_prev;
        pusher_arrays.dx3_prev = species.dx3_prev;
        pusher_arrays.ux1      = species.ux1;
        pusher_arrays.ux2      = species.ux2;
        pusher_arrays.ux3      = species.ux3;
        pusher_arrays.phi      = species.phi;
        pusher_arrays.tag      = species.tag;

        // toggle to indicate whether pgen defines the external force
        bool has_extforce = false;
        if constexpr (arch::traits::pgen::HasExtForce<PG>) {
          has_extforce = true;
          // toggle to indicate whether the ext force applies to current species
          if (::traits::has_member<::traits::species_t, decltype(PG::ext_force)>::value) {
            has_extforce &= std::find(pgen.ext_force.species.begin(),
                                      pgen.ext_force.species.end(),
                                      species.index()) !=
                            pgen.ext_force.species.end();
          }
        }

        pusher_params.ext_force = has_extforce;

        if (not has_atmosphere and not has_extforce) {
          Kokkos::parallel_for("ParticlePusher",
                               species.rangeActiveParticles(),
                               kernel::sr::Pusher_kernel<M>(pusher_params,
                                                            pusher_arrays,
                                                            domain.fields.em,
                                                            domain.mesh.metric));
        } else if (has_atmosphere and not has_extforce) {
          const auto force =
            kernel::sr::Force<M::PrtlDim, M::CoordType, kernel::sr::NoForce_t, true> {
              { gx1, gx2, gx3 },
              x_surf,
              ds
          };
          Kokkos::parallel_for(
            "ParticlePusher",
            species.rangeActiveParticles(),
            kernel::sr::Pusher_kernel<M, decltype(force)>(pusher_params,
                                                          pusher_arrays,
                                                          domain.fields.em,
                                                          domain.mesh.metric,
                                                          force));
        } else if (not has_atmosphere and has_extforce) {
          if constexpr (arch::traits::pgen::HasExtForce<PG>) {
            const auto force =
              kernel::sr::Force<M::PrtlDim, M::CoordType, decltype(pgen.ext_force), false> {
                pgen.ext_force
              };
            Kokkos::parallel_for(
              "ParticlePusher",
              species.rangeActiveParticles(),
              kernel::sr::Pusher_kernel<M, decltype(force)>(pusher_params,
                                                            pusher_arrays,
                                                            domain.fields.em,
                                                            domain.mesh.metric,
                                                            force));
          } else {
            raise::Error("External force not implemented", HERE);
          }
        } else { // has_atmosphere and has_extforce
          if constexpr (arch::traits::pgen::HasExtForce<PG>) {
            const auto force =
              kernel::sr::Force<M::PrtlDim, M::CoordType, decltype(pgen.ext_force), true> {
                pgen.ext_force,
                { gx1, gx2, gx3 },
                x_surf,
                ds
            };
            Kokkos::parallel_for(
              "ParticlePusher",
              species.rangeActiveParticles(),
              kernel::sr::Pusher_kernel<M, decltype(force)>(pusher_params,
                                                            pusher_arrays,
                                                            domain.fields.em,
                                                            domain.mesh.metric,
                                                            force));
          } else {
            raise::Error("External force not implemented", HERE);
          }
        }
      }
    }

  } // namespace srpic
} // namespace ntt

#endif // ENGINES_SRPIC_PARTICLE_PUSHER_H
