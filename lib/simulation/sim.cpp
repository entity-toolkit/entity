#include "sim.h"

namespace ntt {
  void Simulation::setTitle(const std::string _t) {
    _title = _t;
  }
  InputParams g_input_params;
  Simulation g_sim;
}
