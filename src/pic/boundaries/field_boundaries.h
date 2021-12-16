#ifndef PIC_FLDS_BOUNDARIES_H
#define PIC_FLDS_BOUNDARIES_H

#include "global.h"
#include "meshblock.h"

#include <cstddef>

namespace ntt {

  template <Dimension D>
  class FldBC {
  protected:
    Meshblock<D> m_mblock;
    const std::size_t nxI;
    using index_t = typename RealFieldND<D, static_cast<int>(D)>::size_type;

  public:
    FldBC(const Meshblock<D>& m_mblock_, const std::size_t& nxI_) : m_mblock(m_mblock_), nxI(nxI_) {}
  };

} // namespace ntt

#endif
