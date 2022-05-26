#include "global.h"
#include "pic.h"
#include "pic_filter_currents.hpp"

namespace ntt {
  /**
   * @brief filter currents.
   *
   */
  template <Dimension D>
  void PIC<D>::filterCurrentsSubstep(const real_t&) {
    // Kokkos::parallel_for("", this->m_mblock.loopActiveCells(), ...<D>(this->m_mblock));
  }
} // namespace ntt

template class ntt::PIC<ntt::Dimension::ONE_D>;
template class ntt::PIC<ntt::Dimension::TWO_D>;
template class ntt::PIC<ntt::Dimension::THREE_D>;
