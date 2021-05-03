#ifndef SIMULATION_SIM_H
#define SIMULATION_SIM_H

#include "global.h"

#include <string>

// there are 3 global objects that interact with each other: 
// 1. input params -- all the user defined quantities 
// 2. simulation -- all the logistics (alloc/dealloc, loop, output, send/recv etc)
// 3. meshblock -- all the data

namespace ntt {
  class InputParams {
    // this will be a map
  };

  class Simulation {
  private:
    std::string _title;
  public:
    const std::string& title = _title;
    void setTitle(const std::string _t);
    const std::size_t precision = sizeof(real_t);
  };

  extern InputParams g_input_params;
  extern Simulation g_sim;
}

#endif
