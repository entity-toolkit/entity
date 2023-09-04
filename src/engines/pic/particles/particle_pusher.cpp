#include "particle_pusher.hpp"

#include "wrapper.h"

#include "pic.h"

namespace ntt {

  template <Dimension D>
  void PIC<D>::ParticlesPush(const real_t& factor) {
    auto& mblock = this->meshblock;
    auto& pgen   = this->problem_generator;
    auto  params = *(this->params());
    auto  time   = this->m_time;
    for (auto& species : mblock.particles) {
      auto pusher = species.pusher();
      if (species.npart() == 0 || pusher == ParticlePusher::NONE) {
        continue;
      }
      if (pusher == ParticlePusher::PHOTON) {
        PushLoop<D, Photon_t>(params, mblock, species, pgen, time, factor);
      } else if (pusher == ParticlePusher::BORIS) {
        PushLoop<D, Boris_t>(params, mblock, species, pgen, time, factor);
      } else if (pusher == ParticlePusher::VAY) {
        PushLoop<D, Vay_t>(params, mblock, species, pgen, time, factor);
      } else if (pusher == ParticlePusher::BORIS_GCA) {
        PushLoop<D, Boris_GCA_t>(params, mblock, species, pgen, time, factor);
      } else if (pusher == ParticlePusher::VAY_GCA) {
        PushLoop<D, Vay_GCA_t>(params, mblock, species, pgen, time, factor);
      } else {
        NTTHostError("not implemented");
      }
    }
    NTTLog();
  }

} // namespace ntt

template void ntt::PIC<ntt::Dim1>::ParticlesPush(const real_t&);
template void ntt::PIC<ntt::Dim2>::ParticlesPush(const real_t&);
template void ntt::PIC<ntt::Dim3>::ParticlesPush(const real_t&);