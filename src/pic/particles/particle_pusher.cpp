#include "particle_pusher.hpp"

#include "wrapper.h"

#include "pic.h"

namespace ntt {
  template <Dimension D>
  void PIC<D>::ParticlesPush(const real_t& factor) {
    auto& mblock = this->meshblock;
    auto  params = *(this->params());
    for (auto& species : mblock.particles) {
      const real_t dt { factor * mblock.timestep() };
      const real_t coeff { (species.charge() / species.mass()) * HALF * dt / params.larmor0() };
      Pusher_kernel<D> pusher(mblock, species, coeff, dt);
      pusher.apply();
    }
    NTTLog();
  }

}    // namespace ntt

template void ntt::PIC<ntt::Dim1>::ParticlesPush(const real_t&);
template void ntt::PIC<ntt::Dim2>::ParticlesPush(const real_t&);
template void ntt::PIC<ntt::Dim3>::ParticlesPush(const real_t&);