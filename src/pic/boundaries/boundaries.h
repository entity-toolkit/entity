#ifndef PIC_BOUNDARIES_H
#define PIC_BOUNDARIES_H

#include "global.h"
#include "meshblock.h"

#include <cstddef>

namespace ntt {

class BC1D {
protected:
  Meshblock1D m_mblock;
  const std::size_t nxI;
  using size_type = NTTArray<real_t*>::size_type;

public:
  BC1D(const Meshblock1D& m_mblock_, const std::size_t& nxI_)
      : m_mblock(m_mblock_), nxI(nxI_) {}
};

class BC2D {
protected:
  Meshblock2D m_mblock;
  const std::size_t nxI;
  using size_type = NTTArray<real_t**>::size_type;

public:
  BC2D(const Meshblock2D& m_mblock_, const std::size_t& nxI_)
      : m_mblock(m_mblock_), nxI(nxI_) {}
};

} // namespace ntt

#endif
