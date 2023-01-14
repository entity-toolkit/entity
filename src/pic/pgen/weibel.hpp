#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "input.h"
#include "meshblock.h"
#include "sim_params.h"

#include "archetypes.hpp"
#include "injector.hpp"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct Drift : public EnergyDistribution<D, S> {
    Drift(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock),
        udrift { readFromInput<real_t>(params.inputdata(), "problem", "udrift", 1.0) } {}
    Inline void operator()(const coord_t<D>&,
                           vec_t<Dim3>& v,
                           const int&   species) const override {
      if (species == 0) {
        v[0] = udrift;
      } else {
        v[0] = -udrift;
      }
    }

  private:
    const real_t udrift;
  };

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params) {}
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override {}
  };

  template <>
  inline void ProblemGenerator<Dim2, TypePIC>::UserInitParticles(
    const SimulationParams& params, Meshblock<Dim2, TypePIC>& mblock) {
    InjectUniform<Dim2, TypePIC, Drift>(params, mblock, { 1, 2 }, params.ppc0() * 0.5);
  }
}    // namespace ntt

#endif
