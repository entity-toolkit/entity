#include "global.h"
#include "pic.h"
#include "particle_pusher.hpp"

namespace ntt {
  template <Dimension D>
  void PIC<D>::ParticlesPush(const real_t& factor) {
    auto& mblock = this->meshblock;
    auto  params = *(this->params());
    for (auto& species : mblock.particles) {
      const real_t dt {factor * mblock.timestep()};
      const real_t coeff {(species.charge() / species.mass()) * HALF * dt / params.larmor0()};
      Pusher_kernel<D> pusher(mblock, species, coeff, dt);
      pusher.apply();
    }
  }

} // namespace ntt

template struct ntt::PIC<ntt::Dim1>;
template struct ntt::PIC<ntt::Dim2>;
template struct ntt::PIC<ntt::Dim3>;