/**
 * @file fields_bc.cpp
 * @brief Absorbing boundary conditions for the currents at rmax (for 2D axisymmetric).
 * @implements: `CurrentsBoundaryConditions` method of the `PIC` class
 * @includes: `currents_bc.hpp
 * @depends: `pic.h`
 *
 * @notes: - Periodic boundary conditions are implemented in `currents_exch.cpp`
 *
 */

#include "wrapper.h"

#include "pic.h"

#include "meshblock/meshblock.h"

#ifndef MINKOWSKI_METRIC
  #include "currents_bc.hpp"
#endif

namespace ntt {
  /**
   * @brief Special boundary conditions on currents
   */
#ifdef MINKOWSKI_METRIC
  template <Dimension D>
  void PIC<D>::CurrentsBoundaryConditions() {}

#else

  template <>
  void PIC<Dim2>::CurrentsBoundaryConditions() {
    // auto& mblock = this->meshblock;
    // if (mblock.boundaries[0][1] == BoundaryCondition::ABSORB) {
    //   auto&      pgen      = this->problem_generator;
    //   auto       params    = *(this->params());
    //   auto       r_absorb  = params.metricParameters()[2];
    //   auto       r_max     = mblock.metric.x1_max;
    //   const auto i1_absorb = (std::size_t)(mblock.metric.x1_Sph2Code(r_absorb));
    //   NTTHostErrorIf(i1_absorb >= mblock.i1_max(),
    //                  "Absorbing layer is too small, consider "
    //                  "increasing r_absorb");
    //   /**
    //    *    . . . . . . . . . . . . .
    //    *    .                       .
    //    *    .                       .
    //    *    .   ^= = = = = = = =^   .
    //    *    .   |* * * * * * * *\   .
    //    *    .   |* * * * * * * *\   .
    //    *    .   |               \   .
    //    *    .   |               \   .
    //    *    .   ^- - - - - - - -^   .
    //    *    .                       .
    //    *    .                       .
    //    *    . . . . . . . . . . . . .
    //    *
    //    */
    //   Kokkos::parallel_for(
    //     "CurrentsBoundaryConditions",
    //     CreateRangePolicy<Dim2>({ i1_absorb, 0 },
    //                             { mblock.i1_max(), mblock.i2_max() }),
    //     AbsorbCurrents_kernel<Dim2>(mblock, pgen, r_absorb, r_max));
    // }
  }

  template <>
  void PIC<Dim1>::CurrentsBoundaryConditions() {
    NTTHostError("not applicable");
  }

  template <>
  void PIC<Dim3>::CurrentsBoundaryConditions() {
    NTTHostError("not implemented");
  }
#endif

} // namespace ntt

#ifdef MINKOWSKI_METRIC
template void ntt::PIC<ntt::Dim1>::CurrentsBoundaryConditions();
template void ntt::PIC<ntt::Dim2>::CurrentsBoundaryConditions();
template void ntt::PIC<ntt::Dim3>::CurrentsBoundaryConditions();
#endif