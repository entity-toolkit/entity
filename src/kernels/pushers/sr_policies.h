/**
 * @file kernels/pushers/sr_policies.h
 * @brief Policy structs and factory functions for configuring SR particle pusher behavior
 * @implements
 *   - kernel::sr::PusherPolicy<>
 *   - kernel::sr::MakePusherPolicyEmission<> -> auto
 *   - kernel::sr::MakePusherPolicy<> -> void
 * @namespaces:
 *   - kernel::sr::
 */

#ifndef KERNELS_PUSHERS_SR_POLICIES_H
#define KERNELS_PUSHERS_SR_POLICIES_H

#include "enums.h"
#include "global.h"

#include "traits/metric.h"
#include "traits/pgen.h"
#include "traits/policies.h"
#include "utils/comparators.h"
#include "utils/param_container.h"

#include "archetypes/emission/bethe_heitler.h"
#include "archetypes/emission/compton.h"
#include "archetypes/emission/synchrotron.h"
#include "framework/parameters/parameters.h"
#include "kernels/pushers/context.h"

#include <cmath>

namespace kernel::sr {

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
      return ::arch::emission::Synchrotron<M>(emitted_species,
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
      return ::arch::emission::Compton<M>(emitted_species,
                                          photon_species,
                                          pusher_ctx.mass,
                                          pusher_ctx.charge,
                                          pusher_ctx.radiative_drag_flags,
                                          domain.index(),
                                          params,
                                          domain.random_pool());
    } else if constexpr (E == ntt::EmissionType::BETHE_HEITLER) {
      const auto electron_species = params.get<spidx_t>(
        "radiation.emission.bethe_heitler.electron_species");
      const auto positron_species = params.get<spidx_t>(
        "radiation.emission.bethe_heitler.positron_species");
      raise::ErrorIf(electron_species == 0u or positron_species == 0u or
                       electron_species > domain.species.size() or
                       positron_species > domain.species.size(),
                     "Invalid electron_species or positron_species for "
                     "Bethe-Heitler emission",
                     HERE);
      raise::ErrorIf(electron_species == positron_species,
                     "Bethe-Heitler electron_species and positron_species "
                     "must be different",
                     HERE);
      auto& electrons = domain.species[electron_species - 1];
      auto& positrons = domain.species[positron_species - 1];
      raise::ErrorIf(not(pusher_ctx.mass > 0.0f),
                     "Species with Bethe-Heitler emission must be massive",
                     HERE);
      // the lepton momenta are drawn in units of m_e c
      raise::ErrorIf(not cmp::AlmostEqual_host(electrons.mass(), 1.0f) or
                       not cmp::AlmostEqual_host(positrons.mass(), 1.0f),
                     "Bethe-Heitler lepton species must have mass = 1 (the "
                     "unit of mass is the electron mass)",
                     HERE);
      // a pair is neutral: the two charges must cancel exactly as stored,
      // or every pair would add the difference to the charge of the grid
      raise::ErrorIf(not(electrons.charge() < 0.0f) or
                       not std::isfinite(electrons.charge()) or
                       not(positrons.charge() == -electrons.charge()),
                     "Bethe-Heitler lepton species must have exactly opposite "
                     "charges (negative for electron_species)",
                     HERE);
      auto& tables = ::arch::emission::bh::GetTables(params, pusher_ctx.dt);
      return ::arch::emission::BetheHeitler<M>(electrons,
                                               positrons,
                                               electron_species,
                                               positron_species,
                                               pusher_ctx.mass,
                                               pusher_ctx.dt,
                                               domain.index(),
                                               params,
                                               tables,
                                               domain.random_pool());
    } else {
      raise::Error("Invalid emission type for MakeEmissionPolicy", HERE);
      return ::traits::emission::NoPolicy_t {};
    }
  }

  // The optional-method dispatch below is split into these free function
  // templates on purpose. `if constexpr` whose condition depends only on a
  // template parameter captured from an *enclosing* scope is not reliably
  // discarded inside a generic lambda: several compilers (nvcc, intel, ...)
  // still parse/instantiate the dead branch, which fails when the pgen does
  // not define the optional method. Making the predicate a template parameter
  // of a regular function template guarantees the dead branch is discarded.

  template <bool HasEP, class PGen, class DOM, class Next>
  void DispatchEmissionPolicy(const PGen&          pgen,
                              DOM&                 domain,
                              const PusherContext& pusher_ctx,
                              Next&&               next) {
    if constexpr (HasEP) {
      next(pgen.EmissionPolicy(pusher_ctx.time, pusher_ctx.species_index, domain));
    } else {
      raise::Error("Custom emission policy flag is set but problem "
                   "generator does not define an emission policy",
                   HERE);
    }
  }

  template <bool HasCPU, class PGen, class DOM, class Next>
  void DispatchCustomPrtlUpdate(const PGen&          pgen,
                                DOM&                 domain,
                                const PusherContext& pusher_ctx,
                                Next&&               next) {
    if constexpr (HasCPU) {
      next(pgen.CustomParticleUpdate(pusher_ctx.time,
                                     pusher_ctx.species_index,
                                     domain));
    } else {
      next(::traits::custom_prtl_update::NoPolicy_t {});
    }
  }

  template <bool HasEF, class PGen, class DOM, class Next>
  void DispatchExternalFields(const PGen&          pgen,
                              DOM&                 domain,
                              const PusherContext& pusher_ctx,
                              Next&&               next) {
    if constexpr (HasEF) {
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
  }

  template <MetricClass M, class DOM, class PGen, class F>
  void MakePusherPolicy(const PGen&                  pgen,
                        DOM&                         domain,
                        const ntt::SimulationParams& params,
                        const PusherContext&         pusher_ctx,
                        ntt::EmissionTypeFlag        emission_type,
                        bool                         atm,
                        F&&                          callback) {
    constexpr bool has_emission = ::traits::pgen::HasEmissionPolicy<PGen, DOM>;
    constexpr bool has_cpu = ::traits::pgen::HasCustomPrtlUpdate<PGen, DOM>;
    constexpr bool has_extfields = ::traits::pgen::HasExternalFields<PGen, DOM>;

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
        case ntt::EmissionType::BETHE_HEITLER:
          next(MakePusherPolicyEmission<M, DOM, ntt::EmissionType::BETHE_HEITLER>(
            domain,
            params,
            pusher_ctx));
          break;
        case ntt::EmissionType::CUSTOM:
          DispatchEmissionPolicy<has_emission>(pgen, domain, pusher_ctx, next);
          break;
        case ntt::EmissionType::NONE:
        default:
          next(traits::emission::NoPolicy_t {});
          break;
      }
    };

    auto with_custom_prtl_upd = [&](auto next) {
      DispatchCustomPrtlUpdate<has_cpu>(pgen, domain, pusher_ctx, next);
    };

    auto with_ext_fields = [&](auto next) {
      DispatchExternalFields<has_extfields>(pgen, domain, pusher_ctx, next);
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

} // namespace kernel::sr

#endif // KERNELS_PUSHERS_POLICIES_H
