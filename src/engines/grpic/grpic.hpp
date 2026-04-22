/**
 * @file engines/grpic/grpic.hpp
 * @brief Simulation engine class which specializes on GRPIC
 * @implements
 *   - ntt::GRPICEngine<> : ntt::Engine<>
 * @cpp:
 *   - grpic.cpp
 * @namespaces:
 *   - ntt::
 */

#ifndef ENGINES_GRPIC_GRPIC_HPP
#define ENGINES_GRPIC_GRPIC_HPP

#include "enums.h"
#include "global.h"

#include "traits/metric.h"
#include "utils/numeric.h"
#include "utils/timer.h"

#include "engines/grpic/currents.h"
#include "engines/grpic/fields_bcs.h"
#include "engines/grpic/fieldsolvers.h"
#include "engines/grpic/particle_pusher.h"
#include "engines/grpic/utils.h"
#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"

#include "engines/engine.hpp"
#include "pgen.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <toml11/toml.hpp>

#include <string>

namespace ntt {

  template <GRMetricClass M>
  class GRPICEngine : public Engine<SimEngine::GRPIC, M> {
    using base_t   = Engine<SimEngine::GRPIC, M>;
    using pgen_t   = user::PGen<SimEngine::GRPIC, M>;
    using domain_t = Domain<SimEngine::GRPIC, M>;
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

  public:
    static constexpr auto S { SimEngine::GRPIC };

    GRPICEngine(SimulationParams& params) : base_t { params } {}

    ~GRPICEngine() = default;

    void step_forward(timer::Timers& timers, domain_t& dom) override {
      const auto fieldsolver_enabled = m_params.template get<bool>(
        "algorithms.fieldsolver.enable");
      const auto deposit_enabled = m_params.template get<bool>(
        "algorithms.deposit.enable");
      const auto clear_interval = m_params.template get<std::size_t>(
        "particles.clear_interval");

      if (step == 0) {
        if (fieldsolver_enabled) {
          // communicate fields and apply BCs on the first timestep
          /**
           * Initially: em0::B   --
           *            em0::D   --
           *            em::B    at -1/2
           *            em::D    at -1/2
           *
           *            cur0::J  --
           *            cur::J   --
           *
           *            aux::E   --
           *            aux::H   --
           *
           *            x_prtl   at -1/2
           *            u_prtl   at -1/2
           */

          /**
           * em0::D, em::D, em0::B, em::B <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom,
                                         Comm::B | Comm::B0 | Comm::D | Comm::D0);
          grpic::FieldBoundaries(dom,
                                 m_metadomain.mesh(),
                                 m_pgen,
                                 m_params,
                                 BC::B | BC::D,
                                 grpic::gr_bc::main);

          /**
           * em0::B <- em::B
           * em0::D <- em::D
           *
           * Now: em0::B & em0::D at -1/2
           */
          grpic::CopyFields(dom);

          /**
           * aux::E <- alpha * em::D + beta x em0::B
           * aux::H <- alpha * em::B0 - beta x em::D
           *
           * Now: aux::E & aux::H at -1/2
           */
          grpic::ComputeAuxE(dom, grpic::gr_getE::D_B0);
          grpic::ComputeAuxH(dom, grpic::gr_getH::D_B0);

          /**
           * aux::E, aux::H <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom, Comm::H | Comm::E);
          grpic::FieldBoundaries(dom,
                                 m_metadomain.mesh(),
                                 m_pgen,
                                 m_params,
                                 BC::H | BC::E,
                                 grpic::gr_bc::aux);

          /**
           * em0::B <- (em0::B) <- -curl aux::E
           *
           * Now: em0::B at 0
           */
          grpic::Faraday(dom,
                         m_params,
                         this->engineParams(),
                         grpic::gr_faraday::aux,
                         HALF);

          /**
           * em0::B, em::B <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom, Comm::B | Comm::B0);
          grpic::FieldBoundaries(dom,
                                 m_metadomain.mesh(),
                                 m_pgen,
                                 m_params,
                                 BC::B,
                                 grpic::gr_bc::main);

          /**
           * em::D <- (em0::D) <- curl aux::H
           *
           * Now: em::D at 0
           */
          grpic::Ampere(dom,
                        m_params,
                        this->engineParams(),
                        grpic::gr_ampere::init,
                        HALF);

          /**
           * em0::D, em::D <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom, Comm::D | Comm::D0);
          grpic::FieldBoundaries(dom,
                                 m_metadomain.mesh(),
                                 m_pgen,
                                 m_params,
                                 BC::D,
                                 grpic::gr_bc::main);

          /**
           * aux::E <- alpha * em::D + beta x em0::B
           * aux::H <- alpha * em0::B - beta x em::D
           *
           * Now: aux::E & aux::H at 0
           */
          grpic::ComputeAuxE(dom, grpic::gr_getE::D_B0);
          grpic::ComputeAuxH(dom, grpic::gr_getH::D_B0);

          /**
           * aux::E, aux::H <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom, Comm::H | Comm::E);
          grpic::FieldBoundaries(dom,
                                 m_metadomain.mesh(),
                                 m_pgen,
                                 m_params,
                                 BC::H | BC::E,
                                 grpic::gr_bc::aux);

          // !ADD: GR -- particles?

          /**
           * em0::B <- (em::B) <- -curl aux::E
           *
           * Now: em0::B at 1/2
           */
          grpic::Faraday(dom,
                         m_params,
                         this->engineParams(),
                         grpic::gr_faraday::main,
                         ONE);
          /**
           * em0::B, em::B <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom, Comm::B | Comm::B0);
          grpic::FieldBoundaries(dom,
                                 m_metadomain.mesh(),
                                 m_pgen,
                                 m_params,
                                 BC::B,
                                 grpic::gr_bc::main);

          /**
           * em0::D <- (em0::D) <- curl aux::H
           *
           * Now: em0::D at 1/2
           */
          grpic::Ampere(dom, m_params, this->engineParams(), grpic::gr_ampere::aux, ONE);
          /**
           * em0::D, em::D <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom, Comm::D | Comm::D0);
          grpic::FieldBoundaries(dom,
                                 m_metadomain.mesh(),
                                 m_pgen,
                                 m_params,
                                 BC::D,
                                 grpic::gr_bc::main);

          /**
           * aux::H <- alpha * em0::B - beta x em0::D
           *
           * Now: aux::H at 1/2
           */
          grpic::ComputeAuxH(dom, grpic::gr_getH::D0_B0);
          /**
           * aux::H <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom, Comm::H);
          grpic::FieldBoundaries(dom,
                                 m_metadomain.mesh(),
                                 m_pgen,
                                 m_params,
                                 BC::H,
                                 grpic::gr_bc::aux);

          /**
           * em0::D <- (em::D) <- curl aux::H
           *
           * Now: em0::D at 1
           *      em::D at 0
           */
          grpic::Ampere(dom, m_params, this->engineParams(), grpic::gr_ampere::main, ONE);
          /**
           * em0::D, em::D <- boundary conditions
           */
          m_metadomain.CommunicateFields(dom, Comm::D | Comm::D0);
          grpic::FieldBoundaries(dom,
                                 m_metadomain.mesh(),
                                 m_pgen,
                                 m_params,
                                 BC::D,
                                 grpic::gr_bc::main);

          /**
           * em::D <-> em0::D
           * em::B <-> em0::B
           * em::J <-> em0::J
           */
          grpic::SwapFields(dom);
          /**
           * Finally: em0::B   at -1/2
           *          em0::D   at 0
           *          em::B    at 1/2
           *          em::D    at 1
           *
           *          cur0::J  --
           *          cur::J   --
           *
           *          aux::E   --
           *          aux::H   --
           *
           *          x_prtl   at 1
           *          u_prtl   at 1/2
           */
        } else {
          /**
           * em0::B <- em::B
           * em0::D <- em::D
           *
           * Now: em0::B & em0::D at -1/2
           */
          grpic::CopyFields(dom);
        }
      }

      /**
       * Initially: em0::B   at n-3/2
       *            em0::D   at n-1
       *            em::B    at n-1/2
       *            em::D    at n
       *
       *            cur0::J  --
       *            cur::J   at n-1/2
       *
       *            aux::E   --
       *            aux::H   --
       *
       *            x_prtl   at n
       *            u_prtl   at n-1/2
       */

      if (fieldsolver_enabled) {
        timers.start("FieldSolver");
        /**
         * em0::D <- (em0::D + em::D) / 2
         * em0::B <- (em0::B + em::B) / 2
         *
         * Now: em0::D at n-1/2
         *      em0::B at n-1
         */
        grpic::TimeAverageDB(dom);
        /**
         * aux::E <- alpha * em0::D + beta x em::B
         *
         * Now: aux::E at n-1/2
         */
        grpic::ComputeAuxE(dom, grpic::gr_getE::D0_B);
        timers.stop("FieldSolver");

        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::E);
        timers.stop("Communications");
        timers.start("FieldBoundaries");
        /**
         * aux::E <- boundary conditions
         */
        grpic::FieldBoundaries(dom,
                               m_metadomain.mesh(),
                               m_pgen,
                               m_params,
                               BC::E,
                               grpic::gr_bc::aux);
        timers.stop("FieldBoundaries");

        timers.start("FieldSolver");
        /**
         * em0::B <- (em0::B) <- -curl aux::E
         *
         * Now: em0::B at n
         */
        grpic::Faraday(dom, m_params, this->engineParams(), grpic::gr_faraday::aux, ONE);
        timers.stop("FieldSolver");

        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::B | Comm::B0);
        timers.stop("Communications");
        /**
         * em0::B, em::B <- boundary conditions
         */
        timers.start("FieldBoundaries");
        grpic::FieldBoundaries(dom,
                               m_metadomain.mesh(),
                               m_pgen,
                               m_params,
                               BC::B,
                               grpic::gr_bc::main);
        timers.stop("FieldBoundaries");

        timers.start("FieldSolver");
        /**
         * aux::H <- alpha * em0::B - beta x em::D
         *
         * Now: aux::H at n
         */
        grpic::ComputeAuxH(dom, grpic::gr_getH::D_B0);
        timers.stop("FieldSolver");

        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::H);
        timers.stop("Communications");
        timers.start("FieldBoundaries");
        /**
         * aux::H <- boundary conditions
         */
        grpic::FieldBoundaries(dom,
                               m_metadomain.mesh(),
                               m_pgen,
                               m_params,
                               BC::H,
                               grpic::gr_bc::aux);
        timers.stop("FieldBoundaries");
      }

      {
        /**
         * x_prtl, u_prtl <- em::D, em0::B
         *
         * Now: x_prtl at n + 1, u_prtl at n + 1/2
         */
        timers.start("ParticlePusher");
        grpic::ParticlePush(dom, m_params, this->engineParams());
        timers.stop("ParticlePusher");

        /**
         * cur0::J <- current deposition
         *
         * Now: cur0::J at n+1/2
         */
        if (deposit_enabled) {
          timers.start("CurrentDeposit");
          Kokkos::deep_copy(dom.fields.cur0, ZERO);
          grpic::CurrentsDeposit(dom, this->engineParams());
          timers.stop("CurrentDeposit");

          timers.start("Communications");
          m_metadomain.SynchronizeFields(dom, Comm::J);
          m_metadomain.CommunicateFields(dom, Comm::J);
          timers.stop("Communications");

          timers.start("FieldBoundaries");
          grpic::FieldBoundaries(dom,
                                 m_metadomain.mesh(),
                                 m_pgen,
                                 m_params,
                                 BC::J,
                                 grpic::gr_bc::curr);
          timers.stop("FieldBoundaries");

          timers.start("CurrentFiltering");
          grpic::CurrentsFilter(m_metadomain, dom, m_params);
          timers.stop("CurrentFiltering");
        }

        timers.start("Communications");
        m_metadomain.CommunicateParticles(dom);
        timers.stop("Communications");
      }

      if (fieldsolver_enabled) {
        timers.start("FieldSolver");
        if (deposit_enabled) {
          /**
           * cur::J <- (cur0::J + cur::J) / 2
           *
           * Now: cur::J at n
           */
          grpic::TimeAverageJ(dom);
        }
        /**
         * aux::Е <- alpha * em::D + beta x em0::B
         *
         * Now: aux::Е at n
         */
        grpic::ComputeAuxE(dom, grpic::gr_getE::D_B0);
        timers.stop("FieldSolver");

        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::E);
        timers.stop("Communications");
        timers.start("FieldBoundaries");
        /**
         * aux::Е <- boundary conditions
         */
        grpic::FieldBoundaries(dom,
                               m_metadomain.mesh(),
                               m_pgen,
                               m_params,
                               BC::E,
                               grpic::gr_bc::aux);
        timers.stop("FieldBoundaries");

        timers.start("FieldSolver");
        /**
         * em0::B <- (em::B) <- -curl aux::E
         *
         * Now: em0::B at n+1/2
         *      em::B at n-1/2
         */
        grpic::Faraday(dom, m_params, this->engineParams(), grpic::gr_faraday::main, ONE);
        timers.stop("FieldSolver");

        /**
         * em0::B, em::B <- boundary conditions
         */
        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::B | Comm::B0);
        timers.stop("Communications");
        timers.start("FieldBoundaries");
        grpic::FieldBoundaries(dom,
                               m_metadomain.mesh(),
                               m_pgen,
                               m_params,
                               BC::B,
                               grpic::gr_bc::main);
        timers.stop("FieldBoundaries");

        timers.start("FieldSolver");
        /**
         * em0::D <- (em0::D) <- curl aux::H
         *
         * Now: em0::D at n+1/2
         */
        grpic::Ampere(dom, m_params, this->engineParams(), grpic::gr_ampere::aux, ONE);
        timers.stop("FieldSolver");

        if (deposit_enabled) {
          timers.start("FieldSolver");
          /**
           * em0::D <- (em0::D) <- cur::J
           *
           * Now: em0::D at n+1/2
           */
          grpic::AmpereCurrents(dom,
                                m_params,
                                this->engineParams(),
                                grpic::gr_ampere::aux);
          timers.stop("FieldSolver");
        }

        /**
         * em0::D, em::D <- boundary conditions
         */
        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::D | Comm::D0);
        timers.stop("Communications");
        timers.start("FieldBoundaries");
        grpic::FieldBoundaries(dom,
                               m_metadomain.mesh(),
                               m_pgen,
                               m_params,
                               BC::D,
                               grpic::gr_bc::main);
        timers.stop("FieldBoundaries");

        timers.start("FieldSolver");
        /**
         * aux::H <- alpha * em0::B - beta x em0::D
         *
         * Now: aux::H at n+1/2
         */
        grpic::ComputeAuxH(dom, grpic::gr_getH::D0_B0);
        timers.stop("FieldSolver");

        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::H);
        timers.stop("Communications");
        timers.start("FieldBoundaries");
        /**
         * aux::H <- boundary conditions
         */
        grpic::FieldBoundaries(dom,
                               m_metadomain.mesh(),
                               m_pgen,
                               m_params,
                               BC::H,
                               grpic::gr_bc::aux);
        timers.stop("FieldBoundaries");

        timers.start("FieldSolver");
        /**
         * em0::D <- (em::D) <- curl aux::H
         *
         * Now: em0::D at n+1
         *      em::D at n
         */
        grpic::Ampere(dom, m_params, this->engineParams(), grpic::gr_ampere::main, ONE);
        timers.stop("FieldSolver");

        if (deposit_enabled) {
          timers.start("FieldSolver");
          /**
           * em0::D <- (em0::D) <- cur0::J
           *
           * Now: em0::D at n+1
           */
          grpic::AmpereCurrents(dom,
                                m_params,
                                this->engineParams(),
                                grpic::gr_ampere::main);
          timers.stop("FieldSolver");
        }
        timers.start("FieldSolver");
        /**
         * em::D <-> em0::D
         * em::B <-> em0::B
         * cur::J <-> cur0::J
         */
        grpic::SwapFields(dom);
        timers.stop("FieldSolver");

        /**
         * em0::D, em::D <- boundary conditions
         */
        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, Comm::D | Comm::D0);
        timers.stop("Communications");
        timers.start("FieldBoundaries");
        grpic::FieldBoundaries(dom,
                               m_metadomain.mesh(),
                               m_pgen,
                               m_params,
                               BC::D,
                               grpic::gr_bc::main);
        timers.stop("FieldBoundaries");
      }

      timers.start("ParticleSort");
      m_metadomain.SortParticles(time, step, m_params, dom);
      timers.stop("ParticleSort");

      /**
       * Finally: em0::B   at n-1/2
       *          em0::D   at n
       *          em::B    at n+1/2
       *          em::D    at n+1
       *
       *          cur0::J  (at n)
       *          cur::J   at n+1/2
       *
       *          aux::E   (at n+1/2)
       *          aux::H   (at n)
       *
       *          x_prtl   at n+1
       *          u_prtl   at n+1/2
       */
    }
  };
} // namespace ntt

#endif // ENGINES_GRPIC_GRPIC_HPP
