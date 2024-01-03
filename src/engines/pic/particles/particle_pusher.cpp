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
      if (species.npart() == 0 || species.pusher() == ParticlePusher::NONE) {
        continue;
      }
      if (params.extforceEnabled()) {
        PushLoop<D, true>(params, mblock, species, pgen, time, factor);
      } else {
        PushLoop<D, false>(params, mblock, species, pgen, time, factor);
      }
    }
    NTTLog();
  }

} // namespace ntt

template void ntt::PIC<ntt::Dim1>::ParticlesPush(const real_t&);
template void ntt::PIC<ntt::Dim2>::ParticlesPush(const real_t&);
template void ntt::PIC<ntt::Dim3>::ParticlesPush(const real_t&);