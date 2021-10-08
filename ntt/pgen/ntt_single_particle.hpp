#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "global.h"
#include "pgen.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

template <Dimension D>
struct ProblemGenerator : PGen<D> {
  void userInitFields(SimulationParams&, MeshblockND<D>&);
  void userInitParticles(SimulationParams&, MeshblockND<D>&);
};

} // namespace ntt

#endif
