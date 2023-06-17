/**
 * @file currents_filter.cpp
 * @brief Gaussian filtering of the deposited currents `Meshblock::currentFilters` times.
 * @implements: `CurrentsFilter` method of the `GRPIC` class
 * @includes: `currents_filter.hpp
 * @depends: `grpic.h`
 *
 * @notes: - Filter is applied uniformly everywhere, except for the axis ...
 *           ... in 2D axisymmetric simulations. For that we employ ...
 *           ... a special treatment that reflects the particle shape ...
 *           ... from the axis (see Belyaev 2015).
 *
 */

#include "currents_filter.hpp"

#include "wrapper.h"

#include "grpic.h"

#include "io/output.h"

namespace ntt {

  /**
   * @brief filter currents.
   *
   */
  template <Dimension D>
  void GRPIC<D>::CurrentsFilter() {
    auto& mblock = this->meshblock;
    auto  params = *(this->params());
    for (unsigned short i = 0; i < params.currentFilters(); ++i) {
      Kokkos::deep_copy(mblock.buff, mblock.cur);

      range_t<D> range = mblock.rangeActiveCells();
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
      Kokkos::parallel_for("CurrentsFilter", range, CurrentsFilter_kernel<D>(mblock));
      Exchange(GhostCells::currents);
    }
  }
}    // namespace ntt

template void ntt::GRPIC<ntt::Dim2>::CurrentsFilter();
template void ntt::GRPIC<ntt::Dim3>::CurrentsFilter();