#ifndef PIC_PRTL_BOUNDARIES_H
#define PIC_PRTL_BOUNDARIES_H

#include "global.h"
#include "meshblock.h"

#include <cstddef>

namespace ntt {

template <Dimension D>
class PrtlBC {
protected:
  Meshblock<D> m_mblock;
  Particles<D> m_particles
  using index_t = typename RealArrND<D>::size_type;

public:
  PrtlBC(const Meshblock<D>& m_mblock_, const Particles<D>& m_particles_, const std::size_t& nxI_)
      : m_mblock(m_mblock_), m_particles(m_particles_), nxI(nxI_) {}
};

} // namespace ntt

#endif
