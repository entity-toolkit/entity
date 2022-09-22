#include "global.h"
#include "pic.h"
#include "digital_filter.hpp"
// #include "pic_filter_currents.hpp"

namespace ntt {

  /**
   * @brief filter currents.
   *
   */
  template <Dimension D>
  void PIC<D>::filterCurrentsSubstep(const real_t&) {
    DigitalFilter<D> filter(this->m_mblock.cur,
                            this->m_mblock.cur0,
                            this->m_mblock,
                            this->m_sim_params.current_filters());
    filter.apply();
  }
} // namespace ntt

template class ntt::PIC<ntt::Dim1>;
template class ntt::PIC<ntt::Dim2>;
template class ntt::PIC<ntt::Dim3>;
