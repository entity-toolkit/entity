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

    void userBCFields_x1min(SimulationParams&, Meshblock<D>&) {}
    void userBCFields_x1max(SimulationParams&, Meshblock<D>&) {}
  };

} // namespace ntt

#endif
