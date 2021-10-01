#include "global.h"
#include "simulation.h"

#include <plog/Log.h>

namespace ntt {

template <>
void Simulation<One_D>::boundaryConditions(const real_t& time) {
  UNUSED(time);
}

} // namespace ntt
