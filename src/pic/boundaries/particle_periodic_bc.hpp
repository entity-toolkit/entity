#ifndef PIC_PRTL_BOUNDARIES_PERIODIC_H
#define PIC_PRTL_BOUNDARIES_PERIODIC_H

#include "global.h"
#include "particle_boundaries.h"

#include <vector>

namespace ntt {

class PrtlBC1D_Periodic : public PrtlBC<ONE_D> {
public:
  PrtlBC1D_Periodic(const std::vector<real_t>& m_extent_, const Particles<ONE_D>& m_particles_)
      : PrtlBC<ONE_D>{m_extent_, m_particles_} {}
  Inline void operator()(const index_t p) const {
    if (m_particles.m_x1(p) >= m_extent[1]) {
      m_particles.m_x1(p) -= m_extent[1] - m_extent[0];
    } else if (m_particles.m_x1(p) < m_extent[0]) {
      m_particles.m_x1(p) += m_extent[1] - m_extent[0];
    }
  }
};

class PrtlBC2D_Periodic : public PrtlBC<TWO_D> {
public:
  PrtlBC2D_Periodic(const std::vector<real_t>& m_extent_, const Particles<TWO_D>& m_particles_)
      : PrtlBC<TWO_D>{m_extent_, m_particles_} {}
  Inline void operator()(const index_t p) const {
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
};

class PrtlBC3D_Periodic : public PrtlBC<THREE_D> {
public:
  PrtlBC3D_Periodic(const std::vector<real_t>& m_extent_, const Particles<THREE_D>& m_particles_)
      : PrtlBC<THREE_D>{m_extent_, m_particles_} {}
  Inline void operator()(const index_t p) const {
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
};

} // namespace ntt

#endif
