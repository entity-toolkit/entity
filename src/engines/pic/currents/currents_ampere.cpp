/**
 * @file currents_ampere.cpp
 * @brief E^(n+1) = E' - 4 pi * dt * J
 * @implements: `AmpereCurrents` method of the `PIC` class
 * @includes: `currents_ampere.hpp
 * @depends: `pic.h`
 *
 * @notes: - minus sign in the current is included with the `coeff`, ...
 *           ... so the kernel adds `coeff * J`
 *         - charge renormalization is done to keep the charge density ...
 *           ... independent of the resolution and `ppc0`
 *
 */

#include "currents_ampere.hpp"

#include "wrapper.h"

#include "pic.h"

#include "io/output.h"

namespace ntt {

#ifdef MINKOWSKI_METRIC

  /**
   * @brief Add currents to the E-field
   */
  template <Dimension D>
  void PIC<D>::AmpereCurrents() {
    auto&            mblock = this->meshblock;

    auto             params = *(this->params());
    const auto       dt     = mblock.timestep();
    const auto       rho0   = params.larmor0();
    const auto       de0    = params.skindepth0();
    const auto       n0     = params.ppc0() / mblock.metric.min_cell_volume();
    // constant sqrt of det_h is included here ...
    // ... instead of the kernel
    const coord_t<D> dummy { ZERO };
    const auto       sqrt_h = mblock.metric.sqrt_det_h(dummy);
    const auto       coeff  = -dt * rho0 / (sqrt_h * n0 * SQR(de0));
    Kokkos::parallel_for(
      "AmpereCurrents", mblock.rangeActiveCells(), CurrentsAmpere_kernel<D>(mblock, coeff));

    NTTLog();
  }
#else

  /**
   * @brief Add currents to the E-field
   */
  template <Dimension D>
  void PIC<D>::AmpereCurrents() {
    auto&      mblock = this->meshblock;
    auto       params = *(this->params());
    const auto dt     = mblock.timestep();
    const auto rho0   = params.larmor0();
    const auto de0    = params.skindepth0();
    const auto ncells = mblock.Ni1() * mblock.Ni2() * mblock.Ni3();
    // !HOTFIX: this needs to be verified
    const auto volume = ONE / (real_t)ncells;
    const auto n0     = params.ppc0() / volume;
    const auto coeff  = -dt * rho0 / (n0 * SQR(de0));

    range_t<D> range;
    // skip the axis
    if constexpr (D == Dim1) {
      range = CreateRangePolicy<Dim1>({ mblock.i1_min() }, { mblock.i1_max() });
    } else if constexpr (D == Dim2) {
      range = CreateRangePolicy<Dim2>({ mblock.i1_min(), mblock.i2_min() + 1 },
                                      { mblock.i1_max(), mblock.i2_max() });
    } else if constexpr (D == Dim3) {
      range
        = CreateRangePolicy<Dim3>({ mblock.i1_min(), mblock.i2_min() + 1, mblock.i3_min() },
                                  { mblock.i1_max(), mblock.i2_max(), mblock.i3_max() });
    }

    /**
     *    . . . . . . . . . . . . .
     *    .                       .
     *    .                       .
     *    .   ^= = = = = = = =^   .
     *    .   |  * * * * * * *\   .
     *    .   |  * * * * * * *\   .
     *    .   |  * * * * * * *\   .
     *    .   |  * * * * * * *\   .
     *    .   ^- - - - - - - -^   .
     *    .                       .
     *    .                       .
     *    . . . . . . . . . . . . .
     *
     */
    Kokkos::parallel_for("AmpereCurrents-1", range, CurrentsAmpere_kernel<D>(mblock, coeff));
    // do axes separately
    if constexpr (D == Dim2) {
      /**
       *    . . . . . . . . . . . . .
       *    .                       .
       *    .                       .
       *    .   ^= = = = = = = =^   .
       *    .   |*              \*  .
       *    .   |*              \*  .
       *    .   |*              \*  .
       *    .   |*              \*  .
       *    .   ^- - - - - - - -^   .
       *    .                       .
       *    .                       .
       *    . . . . . . . . . . . . .
       *
       */
      Kokkos::parallel_for("AmpereCurrents-2",
                           CreateRangePolicy<Dim1>({ mblock.i1_min() }, { mblock.i1_max() }),
                           CurrentsAmperePoles_kernel<Dim2>(mblock, coeff));
    }

    NTTLog();
  }
#endif
}    // namespace ntt

template void ntt::PIC<ntt::Dim1>::AmpereCurrents();
template void ntt::PIC<ntt::Dim2>::AmpereCurrents();
template void ntt::PIC<ntt::Dim3>::AmpereCurrents();