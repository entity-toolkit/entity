#include "wrapper.h"
#include "pic.h"
#include "meshblock.h"

#ifndef MINKOWSKI_METRIC
#  include "currents_bc.hpp"
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
  void PIC<Dim1>::CurrentsBoundaryConditions() {}

  template <>
  void PIC<Dim2>::CurrentsBoundaryConditions() {
    auto& mblock   = this->meshblock;
    auto& pgen     = this->problem_generator;
    auto  params   = *(this->params());
    auto  r_absorb = params.metricParameters()[2];
    auto  r_max    = mblock.metric.x1_max;
    // !TODO: no need to do all cells
    Kokkos::parallel_for("2d_absorbing bc currs",
                         mblock.rangeActiveCells(),
                         AbsorbCurrents_kernel<Dim2>(mblock, pgen, r_absorb, r_max));
  }

  template <>
  void PIC<Dim3>::CurrentsBoundaryConditions() {
    NTTHostError("not implemented");
  }
#endif

} // namespace ntt

template void ntt::PIC<ntt::Dim1>::CurrentsBoundaryConditions();
template void ntt::PIC<ntt::Dim2>::CurrentsBoundaryConditions();
template void ntt::PIC<ntt::Dim3>::CurrentsBoundaryConditions();