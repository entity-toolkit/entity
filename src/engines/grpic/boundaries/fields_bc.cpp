/**
 * @file fields_bc.cpp
 * @brief Boundary conditions for the fields (for 2D axisymmetric) ...
 *        ... (a) on the axis
 *        ... (b) absorbing boundaries at rmax
 *        ... (c) user-defined field driving
 * @implements: `FieldsBoundaryConditions` method of the `GRPIC` class
 * @includes: `fields_bc.hpp
 * @depends: `grpic.h`
 */

#include "fields_bc.hpp"

#include "wrapper.h"

#include "grpic.h"

namespace ntt {
  template <>
  void GRPIC<Dim2>::FieldsBoundaryConditions(const gr_bc& g) {
    auto& mblock = this->meshblock;
    auto  params = *(this->params());
    NTTHostErrorIf(g != gr_bc::Dfield && g != gr_bc::Bfield,
                   "Wrong option for `g`");

    if (mblock.boundaries[0][0] == BoundaryCondition::OPEN) {
      /**
       * r = rmin (open boundary)
       */
      const auto i1_min { mblock.i1_min() };
      auto       range_x1 = CreateRangePolicy<Dim1>({ mblock.i2_min() },
                                              { mblock.i2_max() + 1 });
      if (g == gr_bc::Dfield) {
        Kokkos::parallel_for(
          "FieldsBoundaryConditions-1",
          range_x1,
          Lambda(index_t i2) {
            mblock.em0(i1_min - 1, i2, em::ex1) = mblock.em0(i1_min, i2, em::ex1);
            mblock.em0(i1_min, i2, em::ex2) = mblock.em0(i1_min + 1, i2, em::ex2);
            mblock.em0(i1_min - 1, i2, em::ex2) = mblock.em0(i1_min, i2, em::ex2);
            mblock.em0(i1_min, i2, em::ex3) = mblock.em0(i1_min + 1, i2, em::ex3);
            mblock.em0(i1_min - 1, i2, em::ex3) = mblock.em0(i1_min, i2, em::ex3);

            mblock.em(i1_min - 1, i2, em::ex1) = mblock.em(i1_min, i2, em::ex1);
            mblock.em(i1_min, i2, em::ex2) = mblock.em(i1_min + 1, i2, em::ex2);
            mblock.em(i1_min - 1, i2, em::ex2) = mblock.em(i1_min, i2, em::ex2);
            mblock.em(i1_min, i2, em::ex3) = mblock.em(i1_min + 1, i2, em::ex3);
            mblock.em(i1_min - 1, i2, em::ex3) = mblock.em(i1_min, i2, em::ex3);
          });
      } else if (g == gr_bc::Bfield) {
        Kokkos::parallel_for(
          "FieldsBoundaryConditions-2",
          range_x1,
          Lambda(index_t i2) {
            mblock.em0(i1_min, i2, em::bx1) = mblock.em0(i1_min + 1, i2, em::bx1);
            mblock.em0(i1_min - 1, i2, em::bx1) = mblock.em0(i1_min, i2, em::bx1);
            mblock.em0(i1_min - 1, i2, em::bx2) = mblock.em0(i1_min, i2, em::bx2);
            mblock.em0(i1_min - 1, i2, em::bx3) = mblock.em0(i1_min, i2, em::bx3);

            mblock.em(i1_min, i2, em::bx1) = mblock.em(i1_min + 1, i2, em::bx1);
            mblock.em(i1_min - 1, i2, em::bx1) = mblock.em(i1_min, i2, em::bx1);
            mblock.em(i1_min - 1, i2, em::bx2) = mblock.em(i1_min, i2, em::bx2);
            mblock.em(i1_min - 1, i2, em::bx3) = mblock.em(i1_min, i2, em::bx3);
          });
      }
    }
    if (mblock.boundaries[0][1] == BoundaryCondition::ABSORB ||
        mblock.boundaries[0][1] == BoundaryCondition::OPEN) {
      /**
       * r = rmax (open boundary)
       */
      const auto i1_max { mblock.i1_max() };
      auto       range_x2 = CreateRangePolicy<Dim1>({ mblock.i2_min() },
                                              { mblock.i2_max() });
      if (g == gr_bc::Dfield) {
        Kokkos::parallel_for(
          "FieldsBoundaryConditions-3",
          range_x2,
          Lambda(index_t i2) {
            mblock.em0(i1_max, i2, em::ex2) = mblock.em0(i1_max - 1, i2, em::ex2);
            mblock.em0(i1_max, i2, em::ex3) = mblock.em0(i1_max - 1, i2, em::ex3);

            mblock.em(i1_max, i2, em::ex2) = mblock.em(i1_max - 1, i2, em::ex2);
            mblock.em(i1_max, i2, em::ex3) = mblock.em(i1_max - 1, i2, em::ex3);
          });
      } else if (g == gr_bc::Bfield) {
        Kokkos::parallel_for(
          "FieldsBoundaryConditions-4",
          range_x2,
          Lambda(index_t i2) {
            mblock.em0(i1_max, i2, em::bx1) = mblock.em0(i1_max - 1, i2, em::bx1);
            mblock.em(i1_max, i2, em::bx1) = mblock.em(i1_max - 1, i2, em::bx1);
          });
      }
    }
    if (mblock.boundaries[0][1] == BoundaryCondition::ABSORB) {
      /**
       * r = rmax (absorbing boundary)
       */
      const auto    r_absorb = params.metricParameters()[2];
      const auto    r_max    = mblock.metric.x1_max;
      coord_t<Dim2> xcu;
      mblock.metric.x_Sph2Code({ r_absorb, 0.0 }, xcu);
      NTTHostErrorIf((std::size_t)(xcu[0]) >= mblock.i1_max() - 1,
                     "Absorbing layer is too small, consider "
                     "increasing r_absorb");
      auto range_absorb = CreateRangePolicy<Dim2>(
        { mblock.i1_min(), mblock.i2_min() },
        { mblock.i1_max() + 1, mblock.i2_max() + 1 });
      if (g == gr_bc::Dfield) {
        Kokkos::parallel_for(
          "FieldsBoundaryConditions-5",
          range_absorb,
          AbsorbDFields_kernel<Dim2>(params, mblock, r_absorb, r_max));
      } else if (g == gr_bc::Bfield) {
        Kokkos::parallel_for(
          "FieldsBoundaryConditions-6",
          range_absorb,
          AbsorbBFields_kernel<Dim2>(params, mblock, r_absorb, r_max));
      }
    }
    if (mblock.boundaries[1][0] == BoundaryCondition::AXIS) {
      const auto i2_min { mblock.i2_min() };
      const auto i2_max { mblock.i2_max() };
      auto       range_x1 = CreateRangePolicy<Dim1>({ mblock.i1_min() - 1 },
                                              { mblock.i1_max() });
      if (g == gr_bc::Bfield) {
        Kokkos::parallel_for(
          "FieldsBoundaryConditions-7",
          range_x1,
          Lambda(index_t i1) {
            mblock.em0(i1, i2_min, em::bx2) = ZERO;
            mblock.em(i1, i2_min, em::bx2)  = ZERO;
          });

        // theta = pi boundary
        Kokkos::parallel_for(
          "FieldsBoundaryConditions-8",
          range_x1,
          Lambda(index_t i1) {
            mblock.em0(i1, i2_max, em::bx2) = ZERO;
            mblock.em(i1, i2_max, em::bx2)  = ZERO;
          });
      }
    }
    NTTLog();
  }

  template <>
  void GRPIC<Dim2>::AuxFieldsBoundaryConditions(const gr_bc& g) {
    auto& mblock = this->meshblock;
    NTTHostErrorIf(g != gr_bc::Efield && g != gr_bc::Hfield,
                   "Wrong option for `g`");
    if (mblock.boundaries[0][0] == BoundaryCondition::OPEN) {
      /**
       * r = rmin (open boundary)
       */
      auto i1_min = mblock.i1_min();
      auto range  = CreateRangePolicy<Dim1>({ mblock.i2_min() },
                                           { mblock.i2_max() + 1 });
      if (g == gr_bc::Efield) {
        Kokkos::parallel_for(
          "2d_bc_rmin",
          range,
          Lambda(index_t i2) {
            mblock.aux(i1_min - 1, i2, em::ex1) = mblock.aux(i1_min, i2, em::ex1);
            mblock.aux(i1_min - 1, i2, em::ex2) = mblock.aux(i1_min, i2, em::ex2);
            mblock.aux(i1_min - 1, i2, em::ex3) = mblock.aux(i1_min, i2, em::ex3);
          });
      } else if (g == gr_bc::Hfield) {
        Kokkos::parallel_for(
          "2d_bc_rmin",
          range,
          Lambda(index_t i2) {
            mblock.aux(i1_min - 1, i2, em::bx1) = mblock.aux(i1_min, i2, em::bx1);
            mblock.aux(i1_min - 1, i2, em::bx2) = mblock.aux(i1_min, i2, em::bx2);
            mblock.aux(i1_min - 1, i2, em::bx3) = mblock.aux(i1_min, i2, em::bx3);
          });
      }
    } else {
      NTTHostErrorIf(mblock.boundaries[0][0] != BoundaryCondition::COMM,
                     "In GRPIC rmin boundaries should always be OPEN");
    }
    NTTLog();
  }

  template <>
  void GRPIC<Dim3>::FieldsBoundaryConditions(const gr_bc&) {
    NTTHostError("not implemented");
  }

  template <>
  void GRPIC<Dim3>::AuxFieldsBoundaryConditions(const gr_bc&) {
    NTTHostError("not implemented");
  }

} // namespace ntt