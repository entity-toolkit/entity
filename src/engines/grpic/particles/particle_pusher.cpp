#include "particle_pusher.hpp"

#include "wrapper.h"

#include "grpic.h"

#include METRIC_HEADER

namespace ntt {
  template <Dimension D>
  void GRPIC<D>::ParticlesPush(const real_t& factor) {
    auto& mblock = this->meshblock;
    auto  params = *(this->params());
    for (auto& species : mblock.particles) {
      if (species.npart() == 0 || species.pusher() == ParticlePusher::NONE) {
        continue;
      }
      PushLoop<D, Metric<D>>(params, mblock, species, factor);
    }
    NTTLog();
  }

} // namespace ntt

template void ntt::GRPIC<ntt::Dim2>::ParticlesPush(const real_t&);
template void ntt::GRPIC<ntt::Dim3>::ParticlesPush(const real_t&);