#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "global.h"
#include "pgen.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

  template <Dimension D>
  struct ProblemGenerator : PGen<D> {
    ProblemGenerator(SimulationParams&);
    ~ProblemGenerator() = default;

    void userInitFields(SimulationParams&, Meshblock<D>&);
    void userBCFields_x1min(SimulationParams&, Meshblock<D>&);
    void userBCFields_x1max(SimulationParams&, Meshblock<D>&);
  };

} // namespace ntt

#endif
