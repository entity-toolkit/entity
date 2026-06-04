#ifndef ENGINES_HYBRID_HYBRID_HPP
#define ENGINES_HYBRID_HYBRID_HPP

#include "enums.h"

#include "traits/metric.h"
#include "utils/timer.h"

#include "engines/hybrid/fields_bcs.h"
#include "engines/hybrid/fieldsolvers.h"
#include "engines/hybrid/particle_pusher.h"

#include "engines/engine.hpp"

namespace ntt {

  template <CartesianMetricClass M>
  class HYBRIDEngine : public Engine<SimEngine::HYBRID, M> {
    using base_t   = Engine<SimEngine::HYBRID, M>;
    using pgen_t   = user::PGen<SimEngine::HYBRID, M>;
    using domain_t = Domain<SimEngine::HYBRID, M>;
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
    static constexpr auto S { SimEngine::HYBRID };

    HYBRIDEngine(const SimulationParams& params) : base_t { params } {}

    ~HYBRIDEngine() override = default;

    void step_forward(timer::Timers& timers, domain_t& dom) override {
      /**
       * Initially: em::012    --
       *            em::345    Bf^(n)
       *
       *            em0::012   --
       *            em0::345   --
       *
       *            aux::012   V^(n) (except step 0)
       *            aux::3     N^(n) (except step 0)
       *
       *            bckp::012  --
       *            bckp::345  --
       *
       *            cur::012   --
       *
       *            x_prtl   at n
       *            u_prtl   at n
       */
      if (step == 0) {
        // fill Bf^(n) ghosts (periodic / MPI) so the field-solver stencils are valid
        timers.start("Communications");
        m_metadomain.CommunicateFields(dom, ::Comm::EM_345);
        timers.stop("Communications");
        // non-periodic (reflecting-wall) boundaries on the initial Bf, if any
        timers.start("FieldBoundaries");
        hybrid::FieldBoundaries<M::Dim>(dom, m_metadomain.mesh(), BC::B);
        timers.stop("FieldBoundaries");
        // compute N^(0) and V^(0) -> aux::012, aux::3 (deposit-only, no push)
        timers.start("Moments");
        hybrid::DepositMoments(dom, m_params);
        m_metadomain.SynchronizeFields(dom, ::Comm::AUX); // additive remap of deposit tails
        m_metadomain.CommunicateFields(dom, ::Comm::AUX);  // fill ghosts for EMF reads
        timers.stop("Moments");
      }

      // EMF calculation #0 — the un-averaged field E^(n).
      // Using: aux::012 [V^(n)], aux::3 [N^(n)], em::345 [Bf^(n)]
      //
      // Ee^(n) = EMF(N^(n), V^(n), Bf^(n))           -> em::012
      // Ec^(n) = EMF(N^(n), V^(n), Bc^(n)=interp Bf) -> em0::012
      // (these seed the trapezoidal average E' = 1/2(E* + E^(n)) in EMF #1/#2)
      timers.start("FieldSolver");
      hybrid::EMF0(dom, this->engineParams());
      timers.stop("FieldSolver");
      timers.start("Communications");
      m_metadomain.CommunicateFields(dom, ::Comm::EM_012 | ::Comm::EM0_012);
      timers.stop("Communications");

      // Faraday push #1
      // Using: em::012 [Ee^(n)], em::345 [Bf^(n)]
      //
      // Bf* = Bf^(n) + dt * curl Ee^(n)
      //
      // Now: cur::012 <-- Bf*
      timers.start("FieldSolver");
      hybrid::Faraday(dom, this->engineParams(), hybrid::faraday::push1);
      timers.stop("FieldSolver");
      timers.start("Communications");
      m_metadomain.CommunicateFields(dom, ::Comm::CUR); // fill Bf* ghosts for EMF #1
      timers.stop("Communications");

      // EMF calculation #1
      // Using:
      //   aux::012 [P^(n)]
      //   aux::3 [N^(n)]
      //   em::012 [Ee^(n)]
      //   em::345 [Bf^(n)]
      //   em0::012 [Ec^(n)]
      //   cur::012 [Bf*]
      //
      // Bc* = interpolate Bf*
      // Bc^(n) = interpolate Bf^(n)
      // Ee* = EMF(N^(n), P^(n), Bf*)
      // Ec* = EMF(N^(n), P^(n), Bc*)
      //
      // Ee' = 0.5 * (Ee* + Ee^(n))
      // Ec' = 0.5 * (Ec* + Ec^(n))
      // Bc' = 0.5 * (Bc* + Bc^(n))
      //
      // Now:
      //   em0::345 <-- Ee'
      //   bckp::012 <-- Ec'
      //   bckp::345 <-- Bc'
      timers.start("FieldSolver");
      hybrid::EMF(dom, this->engineParams());
      timers.stop("FieldSolver");

      // Particle push #1 (predictor) — Pegasus Fig. 2 steps 7+8.
      // Using: bckp::012 [Ec'], bckp::345 [Bc']; transient (no store, no particle BCs).
      // Pushes in registers and deposits predicted moments: aux::012 <-- V', aux::3 <-- N'
      timers.start("Communications");
      // Ee' (em0::345) ghosts for Faraday push #2; Ec'/Bc' (bckp) ghosts for the gather
      m_metadomain.CommunicateFields(dom, ::Comm::EM0_345 | ::Comm::Bckp);
      timers.stop("Communications");
      timers.start("ParticlePusher");
      hybrid::ParticlePush(dom, this->engineParams(), m_params, /* corrector */ false);
      timers.stop("ParticlePusher");
      timers.start("Moments");
      m_metadomain.SynchronizeFields(dom, ::Comm::AUX); // additive remap of deposit tails
      m_metadomain.CommunicateFields(dom, ::Comm::AUX);  // fill ghosts for EMF #2
      timers.stop("Moments");

      // Faraday push #2
      // Using: em0::345 [Ee'], em::345 [Bf^(n)]
      //
      // Bf** = Bf^(n) + dt * curl Ee'
      //
      // Now: cur::012 <-- Bf**
      timers.start("FieldSolver");
      hybrid::Faraday(dom, this->engineParams(), hybrid::faraday::push2);
      timers.stop("FieldSolver");
      timers.start("Communications");
      m_metadomain.CommunicateFields(dom, ::Comm::CUR); // fill Bf** ghosts for EMF #2
      timers.stop("Communications");

      // EMF calculation #2
      // Using:
      //   aux::012 [P']
      //   aux::3 [N']
      //   em::012 [Ee^(n)]
      //   em::345 [Bf^(n)]
      //   em0::012 [Ec^(n)]
      //   cur::012 [Bf**]
      //
      // Bc** = interpolate Bf**
      // Bc^(n) = interpolate Bf^(n)
      // Ee** = EMF(N', P', Bf**)
      // Ec** = EMF(N', P', Bc**)
      //
      // Ee'' = 0.5 * (Ee** + Ee^(n))
      // Ec'' = 0.5 * (Ec** + Ec^(n))
      // Bc'' = 0.5 * (Bc** + Bc^(n))
      //
      // Now:
      //   em0::345 <-- Ee''
      //   bckp:012 <-- Ec''
      //   bckp:345 <-- Bc''
      timers.start("FieldSolver");
      hybrid::EMF(dom, this->engineParams());
      timers.stop("FieldSolver");
      timers.start("Communications");
      // Ee'' (em0::345) ghosts for Faraday push #3; Ec''/Bc'' (bckp) ghosts for the gather
      m_metadomain.CommunicateFields(dom, ::Comm::EM0_345 | ::Comm::Bckp);
      timers.stop("Communications");

      // Faraday push #3
      // Using: em0::345 [Ee''], em::345 [Bf^(n)]
      //
      // Bf^(n+1) = Bf^(n) + dt * curl Ee''
      //
      // Now:
      //   em::345 <-- Bf^(n+1)
      timers.start("FieldSolver");
      hybrid::Faraday(dom, this->engineParams(), hybrid::faraday::push3);
      timers.stop("FieldSolver");
      timers.start("Communications");
      m_metadomain.CommunicateFields(dom, ::Comm::EM_345); // Bf^(n+1) ghosts for next step
      timers.stop("Communications");
      // non-periodic (reflecting-wall) boundaries on Bf^(n+1), if any
      timers.start("FieldBoundaries");
      hybrid::FieldBoundaries<M::Dim>(dom, m_metadomain.mesh(), BC::B);
      timers.stop("FieldBoundaries");

      // Particle push #2 (corrector) — Pegasus Fig. 2 step 12.
      // Using: bckp::012 [Ec''], bckp::345 [Bc'']; accepted (store-back + particle BCs).
      // Pushes and deposits final moments: x_prtl at n+1, aux::012 <-- V^(n+1), aux::3 <-- N^(n+1)
      // (bckp ghosts already filled after EMF #2; Faraday push #3 does not touch bckp)
      timers.start("ParticlePusher");
      hybrid::ParticlePush(dom, this->engineParams(), m_params, /* corrector */ true);
      timers.stop("ParticlePusher");

      timers.start("Moments");
      m_metadomain.SynchronizeFields(dom, ::Comm::AUX); // additive remap of deposit tails (pre-migration)
      m_metadomain.CommunicateFields(dom, ::Comm::AUX);  // fill ghosts for next step's EMF
      timers.stop("Moments");

      timers.start("Communications");
      m_metadomain.CommunicateParticles(dom);
      timers.stop("Communications");

      timers.start("ParticleSort");
      m_metadomain.SortParticles(time, step, m_params, dom);
      timers.stop("ParticleSort");

      /**
       * Finally:   em::012    --
       *            em::345    Bf^(n+1)
       *
       *            em0::012   --
       *            em0::345   --
       *
       *            aux::012   V^(n+1)
       *            aux::3     N^(n+1)
       *
       *            bckp::012  --
       *            bckp::345  --
       *
       *            cur::012   --
       *
       *            x_prtl   at n+1
       *            u_prtl   at n+1
       */
    }
  };
} // namespace ntt

#endif // ENGINES_HYBRID_HYBRID_HPP
