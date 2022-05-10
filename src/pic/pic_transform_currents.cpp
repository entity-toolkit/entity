#include "global.h"
#include "pic.h"
#include "pic_transform_currents.hpp"

namespace ntt {
  /**
   * @brief transform currents.
   *
   */
  template <Dimension D>
  void PIC<D>::transformCurrentsSubstep(const real_t&) {
    Kokkos::parallel_for("transform_currents",
                         this->m_mblock.loopActiveCells(),
                         TransformCurrentsSubstep<D>(this->m_mblock));
  }
} // namespace ntt

template class ntt::PIC<ntt::Dimension::ONE_D>;
template class ntt::PIC<ntt::Dimension::TWO_D>;
template class ntt::PIC<ntt::Dimension::THREE_D>;
