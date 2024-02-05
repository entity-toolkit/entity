/**
 * @file currents_filter.cpp
 * @brief Gaussian filtering of the deposited currents `Meshblock::currentFilters` times.
 * @implements: `CurrentsFilter` method of the `PIC` class
 * @depends: `pic.h`, `kernels/digital_filter.hpp`
 *
 * @notes: - Filter is applied uniformly everywhere, except for the axis ...
 *           ... in 2D axisymmetric simulations. On the axis we reflect ...
 *           ... the shape function.
 *
 */

#include "wrapper.h"

#include "pic.h"

#include "io/output.h"

#include "kernels/digital_filter.hpp"

namespace ntt {

  /**
   * @brief filter currents.
   *
   */
  template <Dimension D>
  void PIC<D>::CurrentsFilter() {
    auto&                   mblock = this->meshblock;
    auto                    params = *(this->params());
    tuple_t<std::size_t, D> size;

    for (short d = 0; d < (short)D; ++d) {
      size[d] = mblock.Ni(d);
    }

    for (unsigned short i = 0; i < params.currentFilters(); ++i) {
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
      this->Communicate(Comm_J);
      Kokkos::deep_copy(mblock.buff, mblock.cur);
      Kokkos::parallel_for("CurrentsFilter",
                           range,
                           DigitalFilter_kernel<D>(mblock.cur, mblock.buff, size));
      WaitAndSynchronize();
    }
    NTTLog();
  }
} // namespace ntt

template void ntt::PIC<ntt::Dim1>::CurrentsFilter();
template void ntt::PIC<ntt::Dim2>::CurrentsFilter();
template void ntt::PIC<ntt::Dim3>::CurrentsFilter();