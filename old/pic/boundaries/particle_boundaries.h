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
    Particles<D> prtls;

  public:
    PrtlBC(const std::vector<real_t>& m_extent_, const Particles<D>& prtls_)
        : m_extent(m_extent_), prtls(prtls_) {}
  };

} // namespace ntt

#endif
