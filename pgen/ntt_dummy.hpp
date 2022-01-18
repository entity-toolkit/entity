#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "global.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    ProblemGenerator(const SimulationParams& sim_params) : PGen<D, S>(sim_params) {}

    void userInitFields(const SimulationParams& sim_params, const Meshblock<D, S>& mblock);
  };

} // namespace ntt

#endif
