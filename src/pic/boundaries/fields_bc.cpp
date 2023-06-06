/**
 * @file fields_bc.cpp
 * @brief Boundary conditions for the fields (for 2D axisymmetric) ...
 *        ... (a) on the axis
 *        ... (b) absorbing boundaries at rmax
 *        ... (c) user-defined field driving
 * @implements: `FieldsBoundaryConditions` method of the `PIC` class
 * @includes: `fields_bc.hpp
 * @depends: `pic.h`
 *
 * @notes: - Periodic boundary conditions are implemented in `fields_exch.cpp`
 *
 */

#include "wrapper.h"

#include "pic.h"

#ifndef MINKOWSKI_METRIC
#  include "fields_bc.hpp"
#endif

namespace ntt {
  /**
   * @brief Special boundary conditions for fields.
   */
#ifdef MINKOWSKI_METRIC
  template <Dimension D>
  void PIC<D>::FieldsBoundaryConditions() {}

#else

  template <>
  void PIC<Dim2>::FieldsBoundaryConditions() {
    /* ----------------------- axisymmetric spherical grid ---------------------- */
    auto& pgen   = this->problem_generator;
    auto& mblock = this->meshblock;
    auto  params = *(this->params());
    if (mblock.boundaries[0][0] == BoundaryCondition::CUSTOM) {
      /**
       * r = rmin (custom boundary, e.g. conductor)
       */
      pgen.UserDriveFields(this->m_time, params, mblock);
    } else {
      NTTHostError("2d boundary condition in r_min have to be CUSTOM");
    }
    if (mblock.boundaries[0][1] == BoundaryCondition::ABSORB
        || mblock.boundaries[0][1] == BoundaryCondition::OPEN) {
      /**
       * r = rmax (open boundary)
       */
      const auto i1_max = mblock.i1_max();
      Kokkos::parallel_for(
        "FieldsBoundaryConditions-1",
        CreateRangePolicy<Dim1>({ mblock.i2_min() }, { mblock.i2_max() }),
        Lambda(index_t i2) {
          mblock.em(i1_max, i2, em::ex2) = mblock.em(i1_max - 1, i2, em::ex2);
          mblock.em(i1_max, i2, em::ex3) = mblock.em(i1_max - 1, i2, em::ex3);
          mblock.em(i1_max, i2, em::bx1) = mblock.em(i1_max - 1, i2, em::bx1);
        });
    }
    if (mblock.boundaries[0][1] == BoundaryCondition::ABSORB) {
      /**
       * r = rmax (absorbing boundary)
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
       */
      const auto    r_absorb = params.metricParameters()[2];
      const auto    r_max    = mblock.metric.x1_max;
      coord_t<Dim2> xcu;
      mblock.metric.x_Sph2Code({ r_absorb, 0.0 }, xcu);
      const auto i1_absorb = (int)(xcu[0]);
      NTTHostErrorIf(i1_absorb >= mblock.i1_max(),
                     "Absorbing layer is too small, consider "
                     "increasing r_absorb");
      Kokkos::parallel_for(
        "FieldsBoundaryConditions-2",
        CreateRangePolicy<Dim2>({ i1_absorb, 0 }, { mblock.i1_max(), mblock.i2_max() }),
        AbsorbFields_kernel<Dim2>(params, mblock, r_absorb, r_max));
    }

    if (mblock.boundaries[1][0] == BoundaryCondition::AXIS) {
      /**
       * theta = 0 / pi boundaries (axis)
       *    . . . . . . . . . . . . .
       *    .  * *               *  .
       *    .  * *               *  .
       *    .   ^= = = = = = = =^   .
       *    .  *|*              \*  .
       *    .  *|*              \*  .
       *    .  *|*              \*  .
       *    .  *|*              \*  .
       *    .   ^- - - - - - - -^   .
       *    .  * *               *  .
       *    .  * *               *  .
       *    . . . . . . . . . . . . .
       *
       */
      const std::size_t i2_min = mblock.i2_min();    // N_GHOSTS
      const std::size_t i2_max = mblock.i2_max();    // N_GHOSTS + sx2
      Kokkos::parallel_for(
        "FieldsBoundaryConditions-3",
        CreateRangePolicy<Dim1>({ 0 }, { mblock.i1_max() + N_GHOSTS }),
        Lambda(index_t i1) {
          // first active cell (axis):
          mblock.em(i1, i2_min, em::bx2)     = 0.0;
          mblock.em(i1, i2_min, em::ex3)     = 0.0;
          // first ghost cell at end of domain (axis):
          mblock.em(i1, i2_max, em::bx2)     = 0.0;
          mblock.em(i1, i2_max, em::ex3)     = 0.0;

          mblock.em(i1, i2_min - 1, em::bx1) = mblock.em(i1, i2_min, em::bx1);
          mblock.em(i1, i2_min - 1, em::bx3) = mblock.em(i1, i2_min, em::bx3);
          mblock.em(i1, i2_max, em::bx1)     = mblock.em(i1, i2_max - 1, em::bx1);
          mblock.em(i1, i2_max, em::bx3)     = mblock.em(i1, i2_max - 1, em::bx3);

          mblock.em(i1, i2_min - 1, em::ex2) = -mblock.em(i1, i2_min, em::ex2);
          mblock.em(i1, i2_max, em::ex2)     = -mblock.em(i1, i2_max - 1, em::ex2);
        });
    } else {
      NTTHostError("2d boundary condition in theta have to be AXIS");
    }
  }

  template <>
  void PIC<Dim1>::FieldsBoundaryConditions() {
    NTTHostError("not applicable");
  }
  template <>
  void PIC<Dim3>::FieldsBoundaryConditions() {
    NTTHostError("not implemented");
  }
#endif

}    // namespace ntt

#ifdef MINKOWSKI_METRIC
template void ntt::PIC<ntt::Dim1>::FieldsBoundaryConditions();
template void ntt::PIC<ntt::Dim2>::FieldsBoundaryConditions();
template void ntt::PIC<ntt::Dim3>::FieldsBoundaryConditions();
#endif