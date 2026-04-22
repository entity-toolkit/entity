/**
 * @file kernels/pushers/policies.h
 * @brief Policy structs and factory functions for configuring particle pusher behavior
 * @implements
 *   - kernel::PusherPolicy<>
 *   - kernel::MakePusherPolicyEmission<> -> auto
 *   - kernel::MakePusherPolicy<> -> void
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_PUSHERS_POLICIES_H
#define KERNELS_PUSHERS_POLICIES_H

#include "enums.h"
#include "global.h"

#include "traits/metric.h"
#include "traits/policies.h"
#include "utils/comparators.h"
#include "utils/param_container.h"

#include "archetypes/emission.h"
#include "kernels/pushers/context.h"

namespace kernel {

  template <MetricClass            M,
            EmissionPolicyClass<M> E = ::traits::emission::NoPolicy_t,
            CustomParticleUpdatePolicyClass<M> CPU = ::traits::custom_prtl_update::NoPolicy_t,
            ExtFieldsPolicyClass<M::Dim> F   = ::traits::extfields::NoPolicy_t,
            bool                         Atm = false>
  struct PusherPolicy {
    using EmissionPolicy                  = E;
    using CustomParticleUpdatePolicy      = CPU;
    using ExternalFieldsPolicy            = F;
    static constexpr bool ApplyAtmosphere = Atm;
    E                     emission_policy;
    CPU                   custom_particle_update_policy;
    F                     external_fields_policy;

    PusherPolicy(const E&   emission_policy               = {},
                 const CPU& custom_particle_update_policy = {},
                 const F&   external_fields_policy        = {})
      : emission_policy { emission_policy }
      , custom_particle_update_policy { custom_particle_update_policy }
      , external_fields_policy { external_fields_policy } {}
  };

  template <MetricClass M, class DOM, ntt::EmissionTypeFlag E>
  auto MakePusherPolicyEmission(DOM&                   domain,
                                const prm::Parameters& params,
                                const PusherContext&   pusher_ctx) {
    if constexpr (E == ntt::EmissionType::SYNCHROTRON) {
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
      return ::arch::EmissionSynchrotron<M>(emitted_species,
                                            photon_species,
                                            pusher_ctx.mass,
                                            pusher_ctx.charge,
                                            pusher_ctx.radiative_drag_flags,
                                            domain.index(),
                                            params,
                                            domain.random_pool());
    } else if constexpr (E == ntt::EmissionType::COMPTON) {
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
      return ::arch::EmissionCompton<M>(emitted_species,
                                        photon_species,
                                        pusher_ctx.mass,
                                        pusher_ctx.charge,
                                        pusher_ctx.radiative_drag_flags,
                                        domain.index(),
                                        params,
                                        domain.random_pool());
    } else {
      raise::Error("Invalid emission type for MakeEmissionPolicy", HERE);
      return ::traits::emission::NoPolicy_t {};
    }
  }

  template <MetricClass M, class DOM, class PGen, class F>
  void MakePusherPolicy(const PGen&                  pgen,
                        DOM&                         domain,
                        const ntt::SimulationParams& params,
                        const PusherContext&         pusher_ctx,
                        ntt::EmissionTypeFlag        emission_type,
                        bool                         atm,
                        F&&                          callback) {
    auto with_emission = [&](auto next) {
      switch (emission_type) {
        case ntt::EmissionType::SYNCHROTRON:
          next(MakePusherPolicyEmission<M, DOM, ntt::EmissionType::SYNCHROTRON>(
            domain,
            params,
            pusher_ctx));
          break;
        case ntt::EmissionType::COMPTON:
          next(MakePusherPolicyEmission<M, DOM, ntt::EmissionType::COMPTON>(
            domain,
            params,
            pusher_ctx));
          break;
        case ntt::EmissionType::CUSTOM:
          if constexpr (::traits::pgen::HasEmissionPolicy<PGen, decltype(domain)>) {
            next(pgen.EmissionPolicy(pusher_ctx.time,
                                     pusher_ctx.species_index,
                                     domain));
          } else {
            raise::Error("Custom emission policy flag is set but problem "
                         "generator does not define an emission policy",
                         HERE);
          }
          break;
        case ntt::EmissionType::NONE:
        default:
          next(traits::emission::NoPolicy_t {});
          break;
      }
    };

    auto with_custom_prtl_upd = [&](auto next) {
      if constexpr (::traits::pgen::HasCustomPrtlUpdate<PGen, DOM>) {
        next(pgen.CustomParticleUpdate(pusher_ctx.time,
                                       pusher_ctx.species_index,
                                       domain));
      } else {
        next(::traits::custom_prtl_update::NoPolicy_t {});
      }
    };

    auto with_ext_fields = [&](auto next) {
      if constexpr (::traits::pgen::HasExternalFields<PGen, DOM>) {
        const auto [apply_extfields, external_fields] = pgen.ExternalFields(
          pusher_ctx.time,
          pusher_ctx.species_index,
          domain);
        if (apply_extfields) {
          next(external_fields);
        } else {
          next(::traits::extfields::NoPolicy_t {});
        }
      } else {
        next(::traits::extfields::NoPolicy_t {});
      }
    };

    with_emission([&](auto ep) {
      with_custom_prtl_upd([&](auto cpu) {
        with_ext_fields([&](auto ef) {
          using E   = decltype(ep);
          using CPU = decltype(cpu);
          using EF  = decltype(ef);
          if (atm) {
            callback(PusherPolicy<M, E, CPU, EF, true> { ep, cpu, ef });
          } else {
            callback(PusherPolicy<M, E, CPU, EF, false> { ep, cpu, ef });
          }
        });
      });
    });
  }

} // namespace kernel

#endif // KERNELS_PUSHERS_POLICIES_H