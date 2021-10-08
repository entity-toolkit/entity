#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "global.h"
#include "pgen.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

template <Dimension D>
struct ProblemGenerator : PGen<D> {
  int m_nx1, m_nx2;
  real_t m_amplitude;

  ProblemGenerator(SimulationParams&);
  ~ProblemGenerator() = default;

  void userInitFields(SimulationParams&, Meshblock<D>&);
  void userInitParticles(SimulationParams&, Meshblock<D>&);
};

} // namespace ntt

#endif
