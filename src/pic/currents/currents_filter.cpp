#include "wrapper.h"
#include "currents_filter.hpp"
#include "pic.h"

namespace ntt {

  /**
   * @brief filter currents.
   *
   */
  template <Dimension D>
  void PIC<D>::CurrentsFilter() {
    auto& mblock = this->meshblock;
    auto  params = *(this->params());
    for (unsigned short i = 0; i < params.currentFilters(); ++i) {
      CurrentsExchange();
      Kokkos::deep_copy(mblock.buff, mblock.cur);
      range_t<D> range = mblock.rangeActiveCells();
#ifndef MINKOWSKI_METRIC
      if constexpr (D == Dim2) {
        range = CreateRangePolicy<Dim2>({mblock.i1_min(), mblock.i2_min()},
                                        {mblock.i1_max(), mblock.i2_max() + 1});
      }
#endif
      Kokkos::parallel_for("filter_pass", range, CurrentsFilter_kernel<D>(mblock));
    }
  }
} // namespace ntt

template void ntt::PIC<ntt::Dim1>::CurrentsFilter();
template void ntt::PIC<ntt::Dim2>::CurrentsFilter();
template void ntt::PIC<ntt::Dim3>::CurrentsFilter();