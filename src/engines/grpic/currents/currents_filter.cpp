/**
 * @file currents_filter.cpp
 * @brief Gaussian filtering of the deposited currents `Meshblock::currentFilters` times.
 * @implements: `CurrentsFilter` method of the `GRPIC` class
 * @depends: `grpic.h`, `kernels/digital_filter.hpp
 *
 * @notes: - Filter is applied uniformly everywhere, except for the axis ...
 *           ... in 2D axisymmetric simulations. On the axis we reflect ...
 *           ... the shape function.
 *
 */

#include "wrapper.h"

#include "grpic.h"

#include "io/output.h"

#include "kernels/digital_filter.hpp"

namespace ntt {

  /**
   * @brief filter currents.
   *
   */
  template <Dimension D>
  void GRPIC<D>::CurrentsFilter() {
    auto&                   mblock = this->meshblock;
    auto                    params = *(this->params());
    tuple_t<std::size_t, D> size;

    for (short d = 0; d < (short)D; ++d) {
      size[d] = mblock.Ni(d);
    }

    for (unsigned short i = 0; i < params.currentFilters(); ++i) {
      Kokkos::deep_copy(mblock.buff, mblock.cur0);

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
      Kokkos::deep_copy(mblock.buff, mblock.cur0);
      Kokkos::parallel_for("CurrentsFilter",
                           range,
                           DigitalFilter_kernel<D>(mblock.cur0, mblock.buff, size));
      this->Communicate(Comm_J);
    }
    NTTLog();
  }
} // namespace ntt

template void ntt::GRPIC<ntt::Dim2>::CurrentsFilter();
template void ntt::GRPIC<ntt::Dim3>::CurrentsFilter();