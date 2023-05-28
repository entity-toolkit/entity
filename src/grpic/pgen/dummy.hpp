#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "meshblock/meshblock.h"
#include "sim_params.h"

#include "utils/archetypes.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}
  };

}    // namespace ntt

#endif
