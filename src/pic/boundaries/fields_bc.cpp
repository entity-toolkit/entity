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
  void PIC<Dim1>::FieldsBoundaryConditions() {}

  template <>
  void PIC<Dim2>::FieldsBoundaryConditions() {
    /* ----------------------- axisymmetric spherical grid ---------------------- */
    // r = rmin boundary
    auto& pgen   = this->problem_generator;
    auto& mblock = this->meshblock;
    auto  params = *(this->params());
    if (mblock.boundaries[0] == BoundaryCondition::USER) {
      pgen.UserDriveFields(this->m_time, params, mblock);
    } else {
      NTTHostError("2d non-user boundary condition not implemented for curvilinear");
    }
    const std::size_t i2_min = mblock.i2_min();    // N_GHOSTS
    const std::size_t i2_max = mblock.i2_max();    // N_GHOSTS + sx2
    Kokkos::parallel_for(
      "2d_bc_theta0-Pi",
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

    if (mblock.boundaries[1] == BoundaryCondition::ABSORB) {
      auto          r_absorb = params.metricParameters()[2];
      auto          r_max    = mblock.metric.x1_max;
      coord_t<Dim2> xc { ZERO };
      mblock.metric.x_Sph2Code({ r_absorb, 0.0 }, xc);
      auto range = mblock.rangeCells({
        { IMAX((int)(xc[0]) - 4, 0), 0 },
        { 0, 0 },
      });
      Kokkos::parallel_for(
        "2d_absorbing bc", range, AbsorbFields_kernel<Dim2>(params, mblock, r_absorb, r_max));
    } else {
      NTTHostError(
        "2d axisymmetry requires absorbing boundary conditions at rmax. Currently specified: `"
        + stringifyBoundaryCondition(mblock.boundaries[1]) + "`.");
    }
  }

  template <>
  void PIC<Dim3>::FieldsBoundaryConditions() {
    NTTHostError("not implemented");
  }
#endif

}    // namespace ntt

template void ntt::PIC<ntt::Dim1>::FieldsBoundaryConditions();
template void ntt::PIC<ntt::Dim2>::FieldsBoundaryConditions();
template void ntt::PIC<ntt::Dim3>::FieldsBoundaryConditions();