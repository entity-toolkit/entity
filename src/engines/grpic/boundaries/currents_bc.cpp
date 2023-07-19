/**
 * @file fields_bc.cpp
 * @brief Absorbing boundary conditions for the currents at rmax (for 2D axisymmetric).
 * @implements: `CurrentsBoundaryConditions` method of the `GRPIC` class
 * @includes: `currents_bc.hpp
 * @depends: `grpic.h`
 *
 */

#include "currents_bc.hpp"

#include "wrapper.h"

#include "grpic.h"

#include "meshblock/meshblock.h"

namespace ntt {
  /**
   * @brief Special boundary conditions on currents
   */

  template <>
  void GRPIC<Dim2>::CurrentsBoundaryConditions() {
    auto&         mblock   = this->meshblock;
    auto&         pgen     = this->problem_generator;
    auto          params   = *(this->params());
    auto          r_absorb = params.metricParameters()[2];
    auto          r_max    = mblock.metric.x1_max;
    coord_t<Dim2> xcu;
    mblock.metric.x_Sph2Code({ r_absorb, 0.0 }, xcu);
    const auto i1_absorb = (std::size_t)(xcu[0]);
    NTTHostErrorIf(i1_absorb >= mblock.i1_max(),
                   "Absorbing layer is too small, consider "
                   "increasing r_absorb");
    /**
     *    . . . . . . . . . . . . .
     *    .                       .
     *    .                       .
     *    .   ^= = = = = = = =^   .
     *    .   |* * * * * * * *\   .
     *    .   |* * * * * * * *\   .
     *    .   |               \   .
     *    .   |               \   .
     *    .   ^- - - - - - - -^   .
     *    .                       .
     *    .                       .
     *    . . . . . . . . . . . . .
     *
     */
    Kokkos::parallel_for(
      "CurrentsBoundaryConditions",
      CreateRangePolicy<Dim2>({ i1_absorb, 0 }, { mblock.i1_max(), mblock.i2_max() }),
      AbsorbCurrents_kernel<Dim2>(mblock, pgen, r_absorb, r_max));
  }

  template <>
  void GRPIC<Dim3>::CurrentsBoundaryConditions() {
    NTTHostError("not implemented");
  }

}    // namespace ntt