#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "meshblock.h"
#include "sim_params.h"
#include "wrapper.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}
  };

}    // namespace ntt

#endif
