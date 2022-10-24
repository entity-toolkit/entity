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
      pgen.UserBCFields(this->m_time, params, mblock);
    } else {
      NTTHostError("2d non-user boundary condition not implemented for curvilinear");
    }
    // theta = 0 boundary
    Kokkos::parallel_for(
      "2d_bc_theta0",
      CreateRangePolicy<Dim2>({0, 0}, {mblock.i1_max() + N_GHOSTS, mblock.i2_min() + 1}),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::bx2) = 0.0;
        mblock.em(i, j, em::ex3) = 0.0;
      });
    // theta = pi boundary
    Kokkos::parallel_for(
      "2d_bc_thetaPi",
      CreateRangePolicy<Dim2>({0, mblock.i2_max()},
                              {mblock.i1_max() + N_GHOSTS, mblock.i2_max() + N_GHOSTS}),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::bx2) = 0.0;
        mblock.em(i, j, em::ex3) = 0.0;
      });

    auto r_absorb = params.metricParameters()[2];
    auto r_max    = mblock.metric.x1_max;
    // !TODO: no need to do all cells
    Kokkos::parallel_for("2d_absorbing bc",
                         mblock.rangeActiveCells(),
                         AbsorbFields_kernel<Dim2>(mblock, r_absorb, r_max));
  }

  template <>
  void PIC<Dim3>::FieldsBoundaryConditions() {
    NTTHostError("not implemented");
  }
#endif

} // namespace ntt

template void ntt::PIC<ntt::Dim1>::FieldsBoundaryConditions();
template void ntt::PIC<ntt::Dim2>::FieldsBoundaryConditions();
template void ntt::PIC<ntt::Dim3>::FieldsBoundaryConditions();