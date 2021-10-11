#ifndef PIC_PRTL_BOUNDARIES_H
#define PIC_PRTL_BOUNDARIES_H

#include "global.h"
#include "meshblock.h"

#include <vector>

namespace ntt {

template <Dimension D>
class PrtlBC {
protected:
  std::vector<real_t> m_extent;
  Particles<D> m_particles;

public:
  PrtlBC(const std::vector<real_t>& m_extent_, const Particles<D>& m_particles_)
      : m_extent(m_extent_), m_particles(m_particles_) {}
};

} // namespace ntt

#endif
