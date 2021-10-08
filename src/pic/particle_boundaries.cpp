#include "global.h"
#include "simulation.h"

#include "particle_periodic_bc.hpp"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {

void Simulation1D::particleBoundaryConditions(const real_t& time) {
  UNUSED(time);
}

void Simulation2D::particleBoundaryConditions(const real_t& time) {
  UNUSED(time);
}

void Simulation3D::particleBoundaryConditions(const real_t& time) {
  UNUSED(time);
}

} // namespace ntt
