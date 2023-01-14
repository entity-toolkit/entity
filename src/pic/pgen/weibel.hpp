#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "meshblock.h"
#include "sim_params.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params) {
      udrift_1 = readFromInput<real_t>(params.inputdata(), "problem", "udrift_1", 1.0);
      udrift_2 = readFromInput<real_t>(params.inputdata(), "problem", "udrift_2", -udrift_1);
    }
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&);

  private:
    real_t udrift_1, udrift_2;
  };

  template <>
  inline void ProblemGenerator<Dim2, TypePIC>::UserInitParticles(
    const SimulationParams& params, Meshblock<Dim2, TypePIC>& mblock) {
    InjectUniform<Dim2, TypePIC, Drift>(params, mblock, { 1, 2 }, params.ppc0() * 0.5);
  }

  // template <>
  // inline void ProblemGenerator<Dim1, TypePIC>::UserInitParticles(
  //   const SimulationParams& params, Meshblock<Dim1, TypePIC>& mblock) {}

  // template <>
  // inline void ProblemGenerator<Dim3, TypePIC>::UserInitParticles(
  //   const SimulationParams& params, Meshblock<Dim3, TypePIC>& mblock) {}

}    // namespace ntt

#endif
