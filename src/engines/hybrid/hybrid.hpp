#ifndef ENGINES_HYBRID_HYBRID_HPP
#define ENGINES_HYBRID_HYBRID_HPP

#include "enums.h"

#include "traits/metric.h"
#include "utils/timer.h"

#include "engines/hybrid/fieldsolvers.h"

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
        // compute N^(0) and V^(0) -> aux::012, aux::3
      }
      // EMF calculation #0
      // Using: aux::012 [V^(n)], aux::3 [N^(n)], em::345 [Bf^(n)]
      //
      // Bc^(n) = interpolate Bf^(n)
      // Ee^(n) = EMF(N^(n), V^(n), Bf^(n))
      // Ec^(n) = EMF(N^(n), V^(n), Bc^(n))
      //
      // Now:
      //   em::012 <-- Ee^(n)
      //   em0::012 <-- Ec^(n)

      // Faraday push #1
      // Using: em::012 [Ee^(n)], em::345 [Bf^(n)]
      //
      // Bf* = Bf^(n) + dt * curl Ee^(n)
      //
      // Now: cur::012 <-- Bf*
      hybrid::Faraday(dom, this->engineParams(), hybrid::faraday::push1);

      // EMF calculation #1
      // Using:
      //   aux::012 [V^(n)]
      //   aux::3 [N^(n)]
      //   em::012 [Ee^(n)]
      //   em::345 [Bf^(n)]
      //   em0::012 [Ec^(n)]
      //   cur::012 [Bf*]
      //
      // Bc* = interpolate Bf*
      // Bc^(n) = interpolate Bf^(n)
      // Ee* = EMF(N^(n), V^(n), Bf*)
      // Ec* = EMF(N^(n), V^(n), Bc*)
      //
      // Ee' = 0.5 * (Ee* + Ee^(n))
      // Ec' = 0.5 * (Ec* + Ec^(n))
      // Bc' = 0.5 * (Bc* + Bc^(n))
      //
      // Now:
      //   em0::345 <-- Ee'
      //   bckp::012 <-- Ec'
      //   bckp::345 <-- Bc'

      // Particle push #1
      // Using: bckp::012 [Ec'], bckp::345 [Bc']
      // Now:
      //   aux::012 <-- V'
      //   aux::3 <-- N'

      // Faraday push #2
      // Using: em0::345 [Ee'], em::345 [Bf^(n)]
      //
      // Bf** = Bf^(n) + dt * curl Ee'
      //
      // Now: cur::012 <-- Bf**
      hybrid::Faraday(dom, this->engineParams(), hybrid::faraday::push2);

      // EMF calculation #2
      // Using:
      //   aux::012 [V']
      //   aux::3 [N']
      //   em::012 [Ee^(n)]
      //   em::345 [Bf^(n)]
      //   em0::012 [Ec^(n)]
      //   cur::012 [Bf**]
      //
      // Bc** = interpolate Bf**
      // Bc^(n) = interpolate Bf^(n)
      // Ee** = EMF(N', V', Bf**)
      // Ec** = EMF(N', V', Bc**)
      //
      // Ee'' = 0.5 * (Ee** + Ee^(n))
      // Ec'' = 0.5 * (Ec** + Ec^(n))
      // Bc'' = 0.5 * (Bc** + Bc^(n))
      //
      // Now:
      //   em0::345 <-- Ee''
      //   bckp:012 <-- Ec''
      //   bckp:345 <-- Bc''

      // Faraday push #3
      // Using: em0::345 [Ee''], em::345 [Bf^(n)]
      //
      // Bf^(n+1) = Bf^(n) + dt * curl Ee''
      //
      // Now:
      //   em::345 <-- Bf^(n+1)
      hybrid::Faraday(dom, this->engineParams(), hybrid::faraday::push3);

      // Particle push #2
      // Using: bckp::012 [Ec''], bckp::345 [Bc'']
      // Now:
      //   x_prtl at n+1
      //   u_prtl at n+1/2
      //   aux::012 <-- V^(n+1)
      //   aux::3 <-- N^(n+1)

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
