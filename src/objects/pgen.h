#ifndef OBJECTS_PGEN_H
#define OBJECTS_PGEN_H

#include "global.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

template <Dimension D>
struct PGen {
  PGen() {}
  PGen(SimulationParams&) {}
  void userInitFields(SimulationParams&, Meshblock<D>&) {}
  void userInitParticles(SimulationParams&, Meshblock<D>&) {}
};

} // namespace ntt

#endif
