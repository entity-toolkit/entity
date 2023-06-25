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
    WeibelInit(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock),
        maxwellian { mblock },
        drift_p { params.get<real_t>("problem", "drift_p", 10.0) },
        drift_b { params.get<real_t>("problem", "drift_b", 10.0) },
        temp_p { params.get<real_t>("problem", "temperature_p", 0.0) },
        temp_b { params.get<real_t>("problem", "temperature_b", 0.0) } {}
    Inline void operator()(const coord_t<D>&,
                           vec_t<Dim3>& v,
                           const int&   species) const override {
      if (species == 1) {
        maxwellian(v, temp_p, drift_p, -dir::z);
      } else if (species == 2) {
        maxwellian(v, temp_p, drift_p, -dir::z);
      } else if (species == 3) {
        maxwellian(v, temp_b, drift_b, dir::z);
      } else if (species == 4) {
        maxwellian(v, temp_b, drift_b, dir::z);
      }
    }

  private:
    const Maxwellian<D, S> maxwellian;
    const real_t           drift_p, drift_b, temp_p, temp_b;
  };

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override {}
  };

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitParticles(
    const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    InjectUniform<Dim2, PICEngine, WeibelInit>(params, mblock, { 1, 2 }, params.ppc0() * 0.25);
    InjectUniform<Dim2, PICEngine, WeibelInit>(params, mblock, { 3, 4 }, params.ppc0() * 0.25);
  }
}    // namespace ntt

#endif
