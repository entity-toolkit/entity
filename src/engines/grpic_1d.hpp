/**
 * @file engines/grpic_1d.hpp
 * @brief Simulation engien class which specialized on 1D GRPIC
 * @implements
 *   - ntt::GRPICEngine_1D<> : ntt::Engine<>
 * @cpp:
 *   - grpic_1d.cpp
 * @namespaces:
 *   - ntt::
 */

#ifndef ENGINES_GRPIC_GRPIC_1D_H
#define ENGINES_GRPIC_GRPIC_1D_H

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

#include "metrics/boyer_lindq_tp.h"

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
  class GRPICEngine_1D : public Engine<SimEngine::GRPIC, M> {

    using base_t   = Engine<SimEngine::GRPIC, M>;
    using pgen_t   = user::PGen<SimEngine::GRPIC, M>;
    using domain_t = Domain<SimEngine::GRPIC, M>;
    // constexprs
    using base_t::pgen_is_ok;
    // contents
    using base_t::m_metadomain;
    using base_t::m_params;
    using base_t::m_pgen;
    // methods
    using base_t::init;
    // variables
    using base_t::dt;
    using base_t::max_steps;
    using base_t::runtime;
    using base_t::step;
    using base_t::time;

    auto constexpr D { M::Dim };

  public:
    static constexpr auto S { SimEngine::GRPIC };

    GRPICEngine_1D(SimulationParams& params)
      : Engine<SimEngine::GRPIC, M> { params } {
        raise::ErrorIf(D != Dim::_1D, "GRPICEngine_1D only works in 1D", HERE);
        raise::ErrorIf(M.Lable != "boyer_lindq_tp", "GRPICEngine_1D only works with BoyerLindqTP metric", HERE);
      }

    ~GRPICEngine_1D() = default;

    void step_forward(timer::Timers&, Domain<SimEngine::GRPIC, M>&) override {
        const auto sort_interval = m_params.template get<std::size_t>(
          "particles.sort_interval");

        if (step == 0) {
          m_metadomain.CommunicateFields(dom, Comm::D);
          /**  
           * !CommunicateFields: Special version for 1D GRPIC needed
           */
          ParticleInjector(dom);
        }

        {
          timers.start("ParticlePusher");
          ParticlePush(dom);
          timers.stop("ParticlePusher");

          timers.start("CurrentDeposit");
          Kokkos::deep_copy(dom.fields.cur, ZERO);
          CurrentsDeposit(dom);
          timers.stop("CurrentDeposit");

          timers.start("Communications");
          m_metadomain.SynchronizeFields(dom, Comm::J);
          /**  
           * !SynchronizeFields: Special version for 1D GRPIC needed
           */
          m_metadomain.CommunicateFields(dom, Comm::J);
          timers.stop("Communications");

          timers.start("CurrentFiltering");
          CurrentsFilter(dom);
          /**  
           * !CurrentsFilter: Special version for 1D GRPIC needed
           */
          timers.stop("CurrentFiltering");

          timers.start("Communications");
          if ((sort_interval > 0) and (step % sort_interval == 0)) {
            m_metadomain.CommunicateParticles(dom, &timers);
          }
          timers.stop("Communications");
        }

        {
          timers.start("FieldSolver");
          CurrentsAmpere(dom);
          timers.stop("FieldSolver");

          timers.start("Communications");
          m_metadomain.CommunicateFields(dom, Comm::D | Comm::J);
          timers.stop("Communications");

        }

        {
          timers.start("Injector");
          DischargeInjector(dom);
          timers.stop("Injector");
        }
    }

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
        //  coeff = q / m dt omegaB0
        const auto coeff   = q_ovr_m * dt *
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
            kernel::gr::Pusher_kernel<BoyerLindqTP<Dim::_1D>>(
                                                      domain.fields.em,
                                                      species.i1,
                                                      species.i1_prev,
                                                      species.dx1,
                                                      species.dx1_prev,
                                                      species.px1,
                                                      species.tag,
                                                      domain.mesh.metric,
                                                      coeff, dt,
                                                      domain.mesh.n_active(in::x1),
                                                      1e-2,
                                                      30,
                                                      domain.mesh.n_active(in::x1)));
      }
    }

    void CurrentsDeposit(domain_t& domain) {
      auto scatter_cur = Kokkos::Experimental::create_scatter_view(
        domain.fields.cur);
      for (auto& species : domain.species) {
        logger::Checkpoint(
          fmt::format("Launching currents deposit kernel for %d [%s] : %lu %f",
                      species.index(),
                      species.label().c_str(),
                      species.npart(),
                      (double)species.charge()),
          HERE);
        if (species.npart() == 0 || cmp::AlmostZero(species.charge())) {
          continue;
        }
        // clang-format off
        /**  
           * !DepositCurrents_kernel: Special version for 1D GRPIC needed
           */
        Kokkos::parallel_for("CurrentsDeposit",
                             species.rangeActiveParticles(),
                             kernel::DepositCurrents_kernel<SimEngine::GRPIC, M>(
                               scatter_cur,
                               species.i1,
                               species.i1_prev,
                               species.dx1,
                               species.dx1_prev,
                               species.px1,
                               species.weight,
                               species.tag,
                               domain.mesh.metric,
                               (real_t)(species.charge()),
                               dt));
      }
      Kokkos::Experimental::contribute(domain.fields.cur, scatter_cur);
    }

    void CurrentsAmpere(domain_t& domain) {
      logger::Checkpoint("Launching Ampere kernel for adding currents", HERE);
      const auto q0    = m_params.template get<real_t>("scales.q0");
      const auto n0    = m_params.template get<real_t>("scales.n0");
      const auto B0    = m_params.template get<real_t>("scales.B0");
      const auto coeff = -dt * q0 * n0 / B0;
      // clang-format off
      Kokkos::parallel_for(
        "Ampere",
        domain.mesh.rangeActiveCells(),
        kernel::gr::CurrentsAmpere_kernel_1D<M>(domain.fields.em,
                                                domain.fields.cur,
                                                domain.mesh.metric,
                                                coeff));
    }


}; // class GRPICEngine_1D

} // namespace ntt

#endif // ENGINES_GRPIC_GRPIC_1D_H
