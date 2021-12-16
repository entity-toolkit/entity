#ifndef PIC_PRTL_BOUNDARIES_PERIODIC_H
#define PIC_PRTL_BOUNDARIES_PERIODIC_H

#include "global.h"
#include "particle_boundaries.h"

#include <vector>

namespace ntt {

  template <Dimension D>
  class PrtlBC_Periodic : public PrtlBC<D> {
    using index_t = typename NTTArray<real_t*>::size_type;

  public:
    PrtlBC_Periodic(const std::vector<real_t>& m_extent_, const Particles<D>& m_particles_)
        : PrtlBC<D> {m_extent_, m_particles_} {}
    Inline void operator()(const index_t) const;
  };

  template <>
  Inline void PrtlBC_Periodic<ONE_D>::operator()(const index_t p) const {
    if (m_particles.m_x1(p) >= m_extent[1]) {
      m_particles.m_x1(p) -= m_extent[1] - m_extent[0];
    } else if (m_particles.m_x1(p) < m_extent[0]) {
      m_particles.m_x1(p) += m_extent[1] - m_extent[0];
    }
  }

  template <>
  Inline void PrtlBC_Periodic<TWO_D>::operator()(const index_t p) const {
    if (m_particles.m_x1(p) >= m_extent[1]) {
      m_particles.m_x1(p) -= m_extent[1] - m_extent[0];
    } else if (m_particles.m_x1(p) < m_extent[0]) {
      m_particles.m_x1(p) += m_extent[1] - m_extent[0];
    }
    if (m_particles.m_x2(p) >= m_extent[3]) {
      m_particles.m_x2(p) -= m_extent[3] - m_extent[2];
    } else if (m_particles.m_x2(p) < m_extent[2]) {
      m_particles.m_x2(p) += m_extent[3] - m_extent[2];
    }
  }

  template <>
  Inline void PrtlBC_Periodic<THREE_D>::operator()(const index_t p) const {
    if (m_particles.m_x1(p) >= m_extent[1]) {
      m_particles.m_x1(p) -= m_extent[1] - m_extent[0];
    } else if (m_particles.m_x1(p) < m_extent[0]) {
      m_particles.m_x1(p) += m_extent[1] - m_extent[0];
    }
    if (m_particles.m_x2(p) >= m_extent[3]) {
      m_particles.m_x2(p) -= m_extent[3] - m_extent[2];
    } else if (m_particles.m_x2(p) < m_extent[2]) {
      m_particles.m_x2(p) += m_extent[3] - m_extent[2];
    }
    if (m_particles.m_x3(p) >= m_extent[5]) {
      m_particles.m_x3(p) -= m_extent[5] - m_extent[4];
    } else if (m_particles.m_x3(p) < m_extent[4]) {
      m_particles.m_x3(p) += m_extent[5] - m_extent[4];
    }
  }

} // namespace ntt

template class ntt::PrtlBC_Periodic<ntt::ONE_D>;
template class ntt::PrtlBC_Periodic<ntt::TWO_D>;
template class ntt::PrtlBC_Periodic<ntt::THREE_D>;

#endif
