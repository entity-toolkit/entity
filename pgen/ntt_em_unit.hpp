#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "global.h"
#include "pgen.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    int m_nx1, m_nx2;
    real_t m_amplitude;

    ProblemGenerator(const SimulationParams& sim_params);

    void userInitFields(const SimulationParams& sim_params, Meshblock<D, S>& mblock) override;
  };

} // namespace ntt

#endif
