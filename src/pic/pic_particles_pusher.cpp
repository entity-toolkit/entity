#if SIMTYPE == PIC_SIMTYPE

#  include "global.h"
#  include "pic.h"
#  include "pic_particles_pusher.hpp"

namespace ntt {
  template <Dimension D>
  void PIC<D>::pushParticlesSubstep(const real_t&, const real_t& factor) {
    for (auto& species : this->m_mblock.particles) {
      const real_t dt {factor * this->m_mblock.timestep()};
      const real_t coeff {(species.charge() / species.mass()) * HALF * dt
                          / this->sim_params().larmor0()};
      const real_t dx {(this->m_mblock.metric.x1_max - this->m_mblock.metric.x1_min) / this->m_mblock.metric.nx1};
      Pusher<D> pusher(this->m_mblock, species, coeff, dt / dx);
      pusher.pushParticles();
    }
  }

} // namespace ntt

#  if SIMTYPE == PIC_SIMTYPE
template class ntt::PIC<ntt::Dimension::ONE_D>;
template class ntt::PIC<ntt::Dimension::TWO_D>;
template class ntt::PIC<ntt::Dimension::THREE_D>;
#  endif

#endif