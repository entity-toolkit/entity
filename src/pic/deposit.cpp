#include "global.h"
#include "simulation.h"

#include <plog/Log.h>

namespace ntt {

void Simulation1D::depositSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "1D deposit";
}

void Simulation2D::depositSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "2D deposit";
}

void Simulation3D::depositSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "3D deposit";
}

} // namespace ntt
