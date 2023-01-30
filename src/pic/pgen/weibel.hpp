#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "meshblock.h"
#include "sim_params.h"

#include "archetypes.hpp"
#include "injector.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct Drift : public EnergyDistribution<D, S> {
    Drift(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock),
        udrift1 { params.get<real_t>("problem", "udrift1", -10.0) },
        udrift2 { params.get<real_t>("problem", "udrift2", -10.0) },
        udrift3 { params.get<real_t>("problem", "udrift3", 10.0) },
        udrift4 { params.get<real_t>("problem", "udrift4", 10.0) } {}
    Inline void operator()(const coord_t<D>&,
                           vec_t<Dim3>& v,
                           const int&   species) const override {
      if (species == 1) {
        v[2] = udrift1;
      } else if (species == 2) {
        v[2] = udrift2;
      } else if (species == 3) {
        v[2] = udrift3;
      } else if (species == 4) {
        v[2] = udrift4;
      }
    }

  private:
    const real_t udrift1, udrift2, udrift3, udrift4;
  };

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override {}
  };

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitParticles(
    const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    InjectUniform<Dim2, PICEngine, Drift>(params, mblock, { 1, 2 }, params.ppc0() * 0.25);
    InjectUniform<Dim2, PICEngine, Drift>(params, mblock, { 3, 4 }, params.ppc0() * 0.25);
  }
}    // namespace ntt

#endif
