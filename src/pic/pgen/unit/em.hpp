#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct ProblemGenerator {
    int    m_nx1, m_nx2;
    real_t m_amplitude;

    ProblemGenerator(const SimulationParams&);
    void userInitParticles(const SimulationParams&, Meshblock<D, S>&) {}
    void userInitFields(const SimulationParams&, Meshblock<D, S>&);
  };

} // namespace ntt

#endif
