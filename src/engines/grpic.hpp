/**
 * @file engines/grpic.hpp
 * @brief Simulation engien class which specialized on GRPIC
 * @implements
 *   - ntt::GRPICEngine<> : ntt::Engine<>
 * @cpp:
 *   - srpic.cpp
 * @namespaces:
 *   - ntt::
 */

#ifndef ENGINES_GRPIC_GRPIC_H
#define ENGINES_GRPIC_GRPIC_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/log.h"
#include "utils/numeric.h"
#include "utils/timer.h"

#include "archetypes/particle_injector.h"
#include "framework/domain/domain.h"
#include "framework/parameters.h"

#include "engines/engine.hpp"
#include "kernels/ampere_gr.hpp"
#include "kernels/currents_deposit.hpp"
#include "kernels/digital_filter.hpp"
#include "kernels/fields_bcs.hpp"
#include "kernels/particle_moments.hpp"
#include "kernels/particle_pusher_1D_gr.hpp"
#include "pgen.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <string>
#include <utility>

namespace ntt {

  template <class M>
  class GRPICEngine : public Engine<SimEngine::GRPIC, M> {
    using Engine<SimEngine::GRPIC, M>::m_params;
    using Engine<SimEngine::GRPIC, M>::m_metadomain;

    auto constexpr D { M::Dim };

  public:
    static constexpr auto S { SimEngine::GRPIC };

    GRPICEngine(SimulationParams& params)
      : Engine<SimEngine::GRPIC, M> { params } {}

    ~GRPICEngine() = default;

    void step_forward(timer::Timers&, Domain<SimEngine::GRPIC, M>&) override {
      static_assert(D == Dim::1D, "GRPIC only supports 1D simulations");
      const auto fieldsolver_enabled = m_params.template get<bool>(
        "algorithms.toggles.fieldsolver");
      const auto deposit_enabled = m_params.template get<bool>(
        "algorithms.toggles.deposit");
      const auto sort_interval = m_params.template get<std::size_t>(
        "particles.sort_interval");

      if (step == 0) {
        // communicate fields and apply BCs on the first timestep
        m_metadomain.CommunicateFields(dom, Comm::E);
        FieldBoundaries(dom, BC::E);
        ParticleInjector(dom);
      }

      {
        timers.start("ParticlePusher");
        ParticlePush(dom);
        timers.stop("ParticlePusher");

        if (deposit_enabled) {
          timers.start("CurrentDeposit");
          Kokkos::deep_copy(dom.fields.cur, ZERO);
          CurrentsDeposit(dom);
          timers.stop("CurrentDeposit");

          timers.start("Communications");
          m_metadomain.SynchronizeFields(dom, Comm::J);
          m_metadomain.CommunicateFields(dom, Comm::J);
          timers.stop("Communications");

          timers.start("CurrentFiltering");
          CurrentsFilter(dom);
          timers.stop("CurrentFiltering");
        }

        timers.start("Communications");
        if ((sort_interval > 0) and (step % sort_interval == 0)) {
          m_metadomain.CommunicateParticles(dom, &timers);
        }
        timers.stop("Communications");
      }

      if (fieldsolver_enabled) {

        if (deposit_enabled) {
          timers.start("FieldSolver");
          CurrentsAmpere(dom);
          timers.stop("FieldSolver");
        }

        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::E | Comm::J);
        timers.stop("Communications");

        timers.start("FieldBoundaries");
        FieldBoundaries(dom, BC::E);
        timers.stop("FieldBoundaries");
      }

      {
        timers.start("Injector");
        ParticleInjector(dom);
        timers.stop("Injector");
      }
    }

    void FieldBoundaries(domain_t& domain, BCTags tags) {}

    void ParticleInjector(domain_t& domain, InjTags tags = Inj::None) {}

    void ParticlePush(domain_t& domain) {
      for (auto& species : domain.species) {
        species.set_unsorted();
        logger::Checkpoint(
          fmt::format("Launching particle pusher kernel for %d [%s] : %lu",
                      species.index(),
                      species.label().c_str(),
                      species.npart()),
          HERE);
        if (species.npart() == 0) {
          continue;
        }
        const auto q_ovr_m = species.mass() > ZERO
                                ? species.charge() / species.mass()
                                : ZERO;
        //  coeff = q / m (dt / 2) omegaB0
        const auto coeff   = q_ovr_m * HALF * dt *
                            m_params.template get<real_t>("scales.omegaB0");
        PrtlPusher::type pusher;
        if (species.pusher() == PrtlPusher::FORCEFREE) {
          pusher = PrtlPusher::FORCEFREE;
        } else {
          raise::Fatal("Invalid particle pusher", HERE);
        }
        // clang-format off
        Kokkos::parallel_for(
          "ParticlePusher",
          species.rangeActiveParticles(),
            kernel::gr::Pusher_kernel<FluxSurface<Dim::_1D>>(
                                                      efield,
                                                      i1,
                                                      i1_prev,
                                                      dx1,
                                                      dx1_prev,
                                                      ux1,
                                                      tag,
                                                      metric,
                                                      coeff, dt,
                                                      nx1,
                                                      1e-5,
                                                      15,
                                                      boundaries));
      }
    }

    void CurrentsDeposit(domain_t& domain) {}

    void CurrentsFilter(domain_t& domain) {}

    void CurrentsAmpere(domain_t& domain) {}
};

} // namespace ntt

#endif // ENGINES_GRPIC_GRPIC_H
