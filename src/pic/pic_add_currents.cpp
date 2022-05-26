#include "global.h"
#include "pic.h"
#include "pic_add_currents.hpp"

namespace ntt {
  /**
   * @brief add currents to the E-field
   *
   */
  template <Dimension D>
  void PIC<D>::addCurrentsSubstep(const real_t&) {
    Kokkos::parallel_for("add_currents", this->m_mblock.loopActiveCells(), AddCurrentsSubstep<D>(this->m_mblock));
  }
} // namespace ntt

template class ntt::PIC<ntt::Dimension::ONE_D>;
template class ntt::PIC<ntt::Dimension::TWO_D>;
template class ntt::PIC<ntt::Dimension::THREE_D>;
