#ifndef OBJECTS_SIMULATION_H
#define OBJECTS_SIMULATION_H

#include "global.h"
#include "sim_params.h"

namespace ntt {
  class Simulation {
  private:
    SimulationParams m_sim_params;
  public:
    Simulation(int argc, char *argv[]);
    ~Simulation();
  };
}

#endif
