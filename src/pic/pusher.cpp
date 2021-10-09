#include "global.h"
#include "simulation.h"
#include "particles.h"

#include "boris.hpp"

#include <stdexcept>

namespace ntt {

void Simulation1D::pushParticlesSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "1D pusher";
  for (auto& species : m_meshblock.particles) {
    const real_t coeff {(species.m_charge / species.m_mass) * static_cast<real_t>(0.5)
                        * m_sim_params.m_timestep / m_sim_params.m_larmor0};
    if (species.m_pusher == BORIS_PUSHER) {
      Kokkos::parallel_for("pusher",
                           species.loopParticles(),
                           Boris1D(m_meshblock, species, coeff, m_sim_params.m_timestep));
    } else {
      throw std::logic_error("ERROR: only Boris pusher is implemented");
    }
  }
}
void Simulation2D::pushParticlesSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "2D pusher";
  for (auto& species : m_meshblock.particles) {
    const real_t coeff {(species.m_charge / species.m_mass) * static_cast<real_t>(0.5)
                        * m_sim_params.m_timestep / m_sim_params.m_larmor0};
    if (species.m_pusher == BORIS_PUSHER) {
      Kokkos::parallel_for("pusher",
                           species.loopParticles(),
                           Boris2D(m_meshblock, species, coeff, m_sim_params.m_timestep));
    } else {
      throw std::logic_error("ERROR: only Boris pusher is implemented");
    }
  }
}
void Simulation3D::pushParticlesSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "3D pusher";
  for (auto& species : m_meshblock.particles) {
    const real_t coeff {(species.m_charge / species.m_mass) * static_cast<real_t>(0.5)
                        * m_sim_params.m_timestep / m_sim_params.m_larmor0};
    if (species.m_pusher == BORIS_PUSHER) {
      Kokkos::parallel_for("pusher",
                           species.loopParticles(),
                           Boris3D(m_meshblock, species, coeff, m_sim_params.m_timestep));
    } else {
      throw std::logic_error("ERROR: only Boris pusher is implemented");
    }
  }
}

} // namespace ntt
