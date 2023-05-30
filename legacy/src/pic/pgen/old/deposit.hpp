#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"
#include "sim_params.h"
#include "meshblock/meshblock.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct ProblemGenerator {
    ProblemGenerator(const SimulationParams&) {}

    void        userInitFields(const SimulationParams&, Meshblock<D, S>&);
    void        userInitParticles(const SimulationParams&, Meshblock<D, S>&);
    void        userBCFields(const real_t&, const SimulationParams&, Meshblock<D, S>&);
    Inline auto userTargetField_br_hat(const Meshblock<D, S>&, const coord_t<D>&) const
      -> real_t {
      return ZERO;
    }
  };
} // namespace ntt

#endif
