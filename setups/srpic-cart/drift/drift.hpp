#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "sim_params.h"

#include "meshblock/meshblock.h"

#include "utils/archetypes.hpp"
#include "utils/injector.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct WeibelInit : public EnergyDistribution<D, S> {
    WeibelInit(const SimulationParams& params, const Meshblock<D, S>& mblock) :
      EnergyDistribution<D, S>(params, mblock),
      maxwellian { mblock },
      drift_p { params.get<real_t>("problem", "drift_p", 10.0) },
      temp_p { params.get<real_t>("problem", "temperature_p", 0.0) } {}

    Inline void operator()(const coord_t<D>&,
                           vec_t<Dim3>& v,
                           const int&   species) const override {
      if (species == 1) {
        maxwellian(v, temp_p, drift_p, -dir::x);
      } else if (species == 2) {
        maxwellian(v, temp_p, drift_p, -dir::x);
      }
    }

  private:
    const Maxwellian<D, S> maxwellian;
    const real_t           drift_p,temp_p;
  };

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}

    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override;
  };

  template <Dimension D, SimulationEngine S>
  inline void ProblemGenerator<D, S>::UserInitParticles(
    const SimulationParams& params,
    Meshblock<D, S>&        mblock) {
    InjectUniform<D, S, WeibelInit>(params, mblock, { 1, 2 }, params.ppc0() * 0.5);
  }
} // namespace ntt

#endif
