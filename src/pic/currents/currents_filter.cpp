/**
 * @file currents_filter.cpp
 * @brief Guassian filtering of the deposited currents `Meshblock::currentFilters` times.
 * @implements: `CurrentsFilter` method of the `PIC` class
 * @includes: `currents_filter.hpp
 * @depends: `pic.h`
 *
 * @notes: - Filter is applied uniformly everywhere, except for the axis ...
 *           ... in 2D axisymmetric simulations. For that we employ ...
 *           ... a special treatment that reflects the particle shape ...
 *           ... from the axis (see Belyaev 2015).
 *
 */

#include "currents_filter.hpp"

#include "wrapper.h"

#include "pic.h"

#include "io/output.h"

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
      Kokkos::deep_copy(mblock.buff, mblock.cur);

      range_t<D> range = mblock.rangeActiveCells();
#ifndef MINKOWSKI_METRIC
      /**
       *    . . . . . . . . . . . . .
       *    .                       .
       *    .                       .
       *    .   ^= = = = = = = =^   .
       *    .   |* * * * * * * *\*  .
       *    .   |* * * * * * * *\*  .
       *    .   |* * * * * * * *\*  .
       *    .   |* * * * * * * *\*  .
       *    .   ^- - - - - - - -^   .
       *    .                       .
       *    .                       .
       *    . . . . . . . . . . . . .
       *
       */
      if constexpr (D == Dim2) {
        range = CreateRangePolicy<Dim2>({ mblock.i1_min(), mblock.i2_min() },
                                        { mblock.i1_max(), mblock.i2_max() + 1 });
      }
#endif
      Kokkos::parallel_for("CurrentsFilter", range, CurrentsFilter_kernel<D>(mblock));
      Exchange(GhostCells::currents);
    }
  }
}    // namespace ntt

template void ntt::PIC<ntt::Dim1>::CurrentsFilter();
template void ntt::PIC<ntt::Dim2>::CurrentsFilter();
template void ntt::PIC<ntt::Dim3>::CurrentsFilter();