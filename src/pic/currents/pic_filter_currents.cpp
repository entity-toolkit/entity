#include "wrapper.h"
#include "pic.h"
// #include "current_filter.hpp"
// #include "pic_filter_currents.hpp"

namespace ntt {

  /**
   * @brief filter currents.
   *
   */
  template <Dimension D>
  void PIC<D>::FilterCurrentsSubstep(const real_t&) {
    // auto&            mblock = this->meshblock;
    // auto             params = *(this->params());
    // CurrentFilter<D> filter(mblock.cur, mblock.cur0, mblock, params.currentFilters());
    // filter.apply();
  }
} // namespace ntt

template struct ntt::PIC<ntt::Dim1>;
template struct ntt::PIC<ntt::Dim2>;
template struct ntt::PIC<ntt::Dim3>;
