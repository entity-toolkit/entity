#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "field_macros.h"
#include "input.h"
#include "meshblock.h"
#include "sim_params.h"

#include "archetypes.hpp"
#include "injector.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params) {}

    inline void UserInitParticles(const SimulationParams& params,
                                  Meshblock<D, S>&        mblock) override {}
  };

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitParticles(
    const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    auto& electrons = mblock.particles[0];
    auto& positrons = mblock.particles[1];
    electrons.setNpart(2);
    positrons.setNpart(2);
    Kokkos::parallel_for(
      "init", CreateRangePolicy<Dim1>({ 0 }, { 1 }), Lambda(index_t) {
        init_prtl_2d(mblock, electrons, 0, 1.2, 0.1, 0.0, -0.2, 0.0);
        init_prtl_2d(mblock, positrons, 0, 1.2, 0.1, 0.0, 0.2, 0.0);
        init_prtl_2d(mblock, electrons, 1, 1.2, constant::PI - 0.1, 0.0, 0.2, 0.0);
        init_prtl_2d(mblock, positrons, 1, 1.2, constant::PI - 0.1, 0.0, -0.2, 0.0);
      });
  }

  template <>
  inline void ProblemGenerator<Dim1, PICEngine>::UserInitParticles(
    const SimulationParams&, Meshblock<Dim1, PICEngine>&) {}
  template <>
  inline void ProblemGenerator<Dim3, PICEngine>::UserInitParticles(
    const SimulationParams&, Meshblock<Dim3, PICEngine>&) {}

}    // namespace ntt

#endif