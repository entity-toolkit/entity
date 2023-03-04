#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "meshblock.h"
#include "particle_macros.h"
#include "sim_params.h"

#include "archetypes.hpp"
#include "injector.hpp"

#ifdef NTTINY_ENABLED
#  include "nttiny/api.h"
#endif

#include <map>

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct LangmuirInit : public EnergyDistribution<D, S> {
    LangmuirInit(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock),
        maxwellian { mblock },
        temperature { params.get<real_t>("problem", "temperature", 0.0) },
        amplitude { params.get<real_t>("problem", "amplitude", 0.01) },
        kx { (real_t)(constant::TWO_PI) * (real_t)(params.get<int>("problem", "nx", 1))
             / (mblock.metric.x1_max - mblock.metric.x1_min) } {}
    Inline void operator()(const coord_t<D>& x,
                           vec_t<Dim3>&      v,
                           const int&        species) const override {
      if (species == 1) {
        maxwellian(v, temperature);
        real_t u1 { amplitude * math::sin(x[0] * kx) };
        v[0] += u1;
      } else {
        v[0] = 0.0;
        v[1] = 0.0;
        v[2] = 0.0;
      }
    }

  private:
    const Maxwellian<D, S> maxwellian;
    const real_t           temperature, amplitude, kx;
  };

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}
    inline void UserInitParticles(const SimulationParams& params, Meshblock<D, S>& mblock) {
      InjectUniform<D, PICEngine, LangmuirInit>(params, mblock, { 1, 2 }, params.ppc0() * 0.5);
    }

#ifdef NTTINY_ENABLED
    inline void UserInitBuffers_nttiny(const SimulationParams&,
                                       const Meshblock<D, S>&,
                                       std::map<std::string, nttiny::ScrollingBuffer>& buffers) {
      nttiny::ScrollingBuffer ex;
      buffers.insert({ "Ex", std::move(ex) });
    }

    inline void UserSetBuffers_nttiny(const real_t& time,
                                      const SimulationParams&,
                                      const Meshblock<D, S>& mblock,
                                      std::map<std::string, nttiny::ScrollingBuffer>& buffers) {
      if constexpr (D == Dim2) {
        buffers["Ex"].AddPoint(
          time, mblock.em_h((int)(mblock.Ni1() / 8.0), (int)(mblock.Ni2() / 2.0), em::ex1));
      }
    }
#endif

  };    // struct ProblemGenerator

}    // namespace ntt

#endif
