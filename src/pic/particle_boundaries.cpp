#include "global.h"
#include "simulation.h"

#include "particle_periodic_bc.hpp"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {

void Simulation1D::particleBoundaryConditions(const real_t& time) {
  UNUSED(time);
  for (auto& species : m_meshblock.particles) {
    Kokkos::parallel_for("prtl_bc",
                         species.loopParticles(),
                         PrtlBC1D_Periodic(m_meshblock.m_extent, species));
  }
}

void Simulation2D::particleBoundaryConditions(const real_t& time) {
  UNUSED(time);
  for (auto& species : m_meshblock.particles) {
    Kokkos::parallel_for("prtl_bc",
                         species.loopParticles(),
                         PrtlBC2D_Periodic(m_meshblock.m_extent, species));
  }
}

void Simulation3D::particleBoundaryConditions(const real_t& time) {
  UNUSED(time);
  for (auto& species : m_meshblock.particles) {
    Kokkos::parallel_for("prtl_bc",
                         species.loopParticles(),
                         PrtlBC3D_Periodic(m_meshblock.m_extent, species));
  }
}

} // namespace ntt
