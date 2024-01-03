#include "particle_pusher.hpp"

#include "wrapper.h"

#include "pic.h"
#include PGEN_HEADER
#include METRIC_HEADER

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
        PushLoop<D, Metric<D>, ProblemGenerator<D, PICEngine>, true>(params,
                                                                     mblock,
                                                                     species,
                                                                     pgen,
                                                                     time,
                                                                     factor);
      } else {
        PushLoop<D, Metric<D>, ProblemGenerator<D, PICEngine>, false>(params,
                                                                      mblock,
                                                                      species,
                                                                      pgen,
                                                                      time,
                                                                      factor);
      }
    }
    NTTLog();
  }

} // namespace ntt

template void ntt::PIC<ntt::Dim1>::ParticlesPush(const real_t&);
template void ntt::PIC<ntt::Dim2>::ParticlesPush(const real_t&);
template void ntt::PIC<ntt::Dim3>::ParticlesPush(const real_t&);