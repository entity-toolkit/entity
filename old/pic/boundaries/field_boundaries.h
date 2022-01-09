#ifndef PIC_FLDS_BOUNDARIES_H
#define PIC_FLDS_BOUNDARIES_H

#include "global.h"
#include "meshblock.h"

#include <cstddef>

namespace ntt {

  template <Dimension D>
  class FldBC {
  protected:
    Meshblock<D> mblock;
    const std::size_t nxI;
    using index_t = typename RealFieldND<D, static_cast<int>(D)>::size_type;

  public:
    FldBC(const Meshblock<D>& mblock_, const std::size_t& nxI_) : mblock(mblock_), nxI(nxI_) {}
  };

} // namespace ntt

#endif
