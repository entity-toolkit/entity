#include "global.h"
#include "simulation.h"
#include "particles.h"

#include "pusher.h"

#include <stdexcept>

namespace ntt {

  template <Dimension D>
  void Simulation<D>::pushParticlesSubstep(const real_t& time) {
    UNUSED(time);
    PLOGD << D << "1D pusher";
    for (auto& species : m_meshblock.particles) {
      const real_t coeff {
          (species.m_charge / species.m_mass) * static_cast<real_t>(0.5) * m_sim_params.m_timestep
          / m_sim_params.m_larmor0};
      Pusher<D> particle_pusher(m_meshblock, species, coeff, m_sim_params.m_timestep);
      particle_pusher.pushAllParticles();
    }
  }

} // namespace ntt

template class ntt::Simulation<ntt::ONE_D>;
template class ntt::Simulation<ntt::TWO_D>;
template class ntt::Simulation<ntt::THREE_D>;
