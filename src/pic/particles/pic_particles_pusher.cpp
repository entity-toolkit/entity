#include "global.h"
#include "pic.h"
#include "pic_particles_pusher.hpp"

namespace ntt {
  template <Dimension D>
  void PIC<D>::pushParticlesSubstep(const real_t&, const real_t& factor) {
    for (auto& species : this->m_mblock.particles) {
      const real_t dt {factor * this->mblock()->timestep()};
      const real_t coeff {(species.charge() / species.mass()) * HALF * dt
                          / this->sim_params()->larmor0()};
      Pusher<D>    pusher(this->m_mblock, species, coeff, dt);
      pusher.pushParticles();
    }
  }

} // namespace ntt

template class ntt::PIC<ntt::Dim1>;
template class ntt::PIC<ntt::Dim2>;
template class ntt::PIC<ntt::Dim3>;