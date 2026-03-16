#ifndef ENGINES_SRPIC_PARTICLE_PUSHER_H
#define ENGINES_SRPIC_PARTICLE_PUSHER_H

#include "enums.h"
#include "global.h"

#include "utils/comparators.h"
#include "utils/log.h"
#include "utils/numeric.h"
#include "utils/param_container.h"

#include "metrics/traits.h"

#include "archetypes/traits.h"
#include "engines/srpic/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/grid.h"
#include "framework/parameters/parameters.h"
#include "kernels/emission/compton.hpp"
#include "kernels/emission/emission.hpp"
#include "kernels/emission/synchrotron.hpp"
#include "kernels/particle_pusher_sr.hpp"

namespace ntt {
  namespace srpic {

    template <class M, class F, class PG, bool Atm>
    void CallPusher(Domain<SimEngine::SRPIC, M>&    domain,
                    const SimulationParams&         params,
                    const kernel::sr::PusherParams& pusher_params,
                    kernel::sr::PusherArrays&       pusher_arrays,
                    EmissionTypeFlag                emission_policy_flag,
                    const range_t<Dim::_1D>&        range,
                    const ndfield_t<M::Dim, 6>&     EB,
                    const M&                        metric,
                    const PG&                       pgen,
                    const F&                        external_fields) {
      if (emission_policy_flag == EmissionType::NONE) {
        const auto no_emission = kernel::NoEmissionPolicy_t<SimEngine::SRPIC, M> {};
        Kokkos::parallel_for(
          "ParticlePusher",
          range,
          kernel::sr::Pusher_kernel<M, F, Atm, decltype(no_emission)>(
            pusher_params,
            pusher_arrays,
            EB,
            metric,
            external_fields,
            no_emission));
      } else if (emission_policy_flag == EmissionType::SYNCHROTRON) {
        const auto photon_species = params.get<spidx_t>(
          "radiation.emission.synchrotron.photon_species");
        raise::ErrorIf(photon_species > domain.species.size(),
                       "Invalid photon_species for Synchrotron emission",
                       HERE);
        auto& emitted_species = domain.species[photon_species - 1];
        raise::ErrorIf(not cmp::AlmostZero_host(emitted_species.mass()),
                       "Emitted photon species must have zero mass",
                       HERE);
        raise::ErrorIf(not cmp::AlmostZero_host(emitted_species.charge()),
                       "Emitted photon species must have zero charge",
                       HERE);
        const auto emission_policy = kernel::emission::Synchrotron<M>(
          emitted_species,
          photon_species,
          pusher_params.mass,
          pusher_params.charge,
          pusher_params.radiative_drag_flags,
          pusher_params.dt,
          domain.index(),
          params,
          domain.random_pool());
        Kokkos::parallel_for(
          "ParticlePusher",
          range,
          kernel::sr::Pusher_kernel<M, F, Atm, decltype(emission_policy)>(
            pusher_params,
            pusher_arrays,
            EB,
            metric,
            external_fields,
            emission_policy));
        const auto n_inj = emission_policy.numbers_injected();
        raise::ErrorIf(n_inj.size() != 1,
                       "Synchrotron emission should only inject one species",
                       HERE);
        domain.species[photon_species - 1].set_npart(
          emitted_species.npart() + n_inj[0]);
        domain.species[photon_species - 1].set_counter(
          emitted_species.counter() + n_inj[0]);
      } else if (emission_policy_flag == EmissionType::COMPTON) {
        const auto photon_species = params.get<spidx_t>(
          "radiation.emission.compton.photon_species");
        raise::ErrorIf(photon_species > domain.species.size(),
                       "Invalid photon_species for Compton emission",
                       HERE);
        auto& emitted_species = domain.species[photon_species - 1];
        raise::ErrorIf(not cmp::AlmostZero_host(emitted_species.mass()),
                       "Emitted photon species must have zero mass",
                       HERE);
        raise::ErrorIf(not cmp::AlmostZero_host(emitted_species.charge()),
                       "Emitted photon species must have zero charge",
                       HERE);
        const auto emission_policy = kernel::emission::Compton<M>(
          emitted_species,
          photon_species,
          pusher_params.mass,
          pusher_params.charge,
          pusher_params.radiative_drag_flags,
          pusher_params.dt,
          domain.index(),
          params,
          domain.random_pool());
        Kokkos::parallel_for(
          "ParticlePusher",
          range,
          kernel::sr::Pusher_kernel<M, F, Atm, decltype(emission_policy)>(
            pusher_params,
            pusher_arrays,
            EB,
            metric,
            external_fields,
            emission_policy));
        const auto n_inj = emission_policy.numbers_injected();
        raise::ErrorIf(n_inj.size() != 1,
                       "Compton emission should only inject one species",
                       HERE);
        domain.species[photon_species - 1].set_npart(
          emitted_species.npart() + n_inj[0]);
        domain.species[photon_species - 1].set_counter(
          emitted_species.counter() + n_inj[0]);
      } else if (emission_policy_flag == EmissionType::CUSTOM) {
        if constexpr (
          arch::traits::pgen::HasEmissionPolicy<PG, M, decltype(domain)>) {
          const auto emission_policy = pgen.EmissionPolicy(pusher_params.time,
                                                           pusher_params.species_index,
                                                           domain,
                                                           params);
          static_assert(
            kernel::traits::emission::IsValid<decltype(emission_policy), M>,
            "Custom emission policy does not satisfy the required "
            "interface");
          Kokkos::parallel_for(
            "ParticlePusher",
            range,
            kernel::sr::Pusher_kernel<M, F, Atm, decltype(emission_policy)>(
              pusher_params,
              pusher_arrays,
              EB,
              metric,
              external_fields,
              emission_policy));
          const auto emitted_species = emission_policy.emitted_species_indices();
          const auto n_inj = emission_policy.number_injected();
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
        } else {
          raise::Error("Custom emission policy flag is set but problem "
                       "generator does not define an emission policy",
                       HERE);
        }
      } else {
        raise::Error("Unrecognized emission policy flag", HERE);
      }
    }

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
        pusher_params.species_index        = species.index();
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

        if (has_atmosphere) {
          pusher_params.atmosphere_params.set("gx1", gx1);
          pusher_params.atmosphere_params.set("gx2", gx2);
          pusher_params.atmosphere_params.set("gx3", gx3);
          pusher_params.atmosphere_params.set("x_surf", x_surf);
          pusher_params.atmosphere_params.set("ds", ds);
        }

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

        auto pusher_arrays = species.PusherKernelArrays();

        // toggle to indicate whether pgen defines the external force
        bool has_extfields = false;
        if constexpr (arch::traits::pgen::HasExtFields<PG>) {
          has_extfields = true;
          // toggle to indicate whether the ext force applies to current species
          if (::traits::has_member<::traits::species_t, decltype(PG::ext_fields)>::value) {
            has_extfields &= std::find(pgen.ext_fields.species.begin(),
                                       pgen.ext_fields.species.end(),
                                       species.index()) !=
                             pgen.ext_fields.species.end();
          }
        }

        if (not has_atmosphere and not has_extfields) {
          CallPusher<M, kernel::sr::NoField_t, decltype(pgen), false>(
            domain,
            params,
            pusher_params,
            pusher_arrays,
            species.emission_policy_flag(),
            species.rangeActiveParticles(),
            domain.fields.em,
            domain.mesh.metric,
            pgen,
            kernel::sr::NoField_t {});
        } else if (has_atmosphere and not has_extfields) {
          CallPusher<M, kernel::sr::NoField_t, decltype(pgen), true>(
            domain,
            params,
            pusher_params,
            pusher_arrays,
            species.emission_policy_flag(),
            species.rangeActiveParticles(),
            domain.fields.em,
            domain.mesh.metric,
            pgen,
            kernel::sr::NoField_t {});
        } else if (not has_atmosphere and has_extfields) {
          if constexpr (arch::traits::pgen::HasExtFields<PG>) {
            CallPusher<M, decltype(pgen.ext_fields), decltype(pgen), false>(
              domain,
              params,
              pusher_params,
              pusher_arrays,
              species.emission_policy_flag(),
              species.rangeActiveParticles(),
              domain.fields.em,
              domain.mesh.metric,
              pgen,
              pgen.ext_fields);
          } else {
            raise::Error("External fields not implemented", HERE);
          }
        } else { // has_atmosphere and has_extforce
          if constexpr (arch::traits::pgen::HasExtFields<PG>) {
            CallPusher<M, decltype(pgen.ext_fields), decltype(pgen), true>(
              domain,
              params,
              pusher_params,
              pusher_arrays,
              species.emission_policy_flag(),
              species.rangeActiveParticles(),
              domain.fields.em,
              domain.mesh.metric,
              pgen,
              pgen.ext_fields);
          } else {
            raise::Error("External fields not implemented", HERE);
          }
        }
      }
    }

  } // namespace srpic
} // namespace ntt

#endif // ENGINES_SRPIC_PARTICLE_PUSHER_H
