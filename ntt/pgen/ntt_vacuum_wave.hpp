#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "global.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

// TODO!: we need to have a parent class with default methods
class ProblemGenerator {
  int m_nx1, m_nx2;
  real_t m_amplitude;

public:
  ProblemGenerator(SimulationParams& sim_params);
  ~ProblemGenerator() = default;

  template <template <typename T> class D>
  void userInitFields(SimulationParams& sim_params, Meshblock<D>& mblock);
};

} // namespace ntt

#endif
