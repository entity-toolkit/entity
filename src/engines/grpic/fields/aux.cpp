/**
 * @file aux.cpp
 * @brief auxiliary functions for compute intermediate fields
 * @implements: `ComputeAuxE`, `ComputeAuxH`, `TimeAverageDB`, `TimeAverageJ` methods of the `GRPIC` class
 * @includes: `kernels/aux_fields_gr.hpp`
 * @depends: `grpic.h`
 *
 */

#include "wrapper.h"

#include "grpic.h"

#include "kernels/aux_fields_gr.hpp"
#include METRIC_HEADER

namespace ntt {

  template <>
  void GRPIC<Dim2>::ComputeAuxE(const gr_getE& g) {
    auto& mblock = this->meshblock;
    auto range = CreateRangePolicy<Dim2>({ mblock.i1_min() - 1, mblock.i2_min() },
                                         { mblock.i1_max(), mblock.i2_max() + 1 });
    if (g == gr_getE::D0_B) {
      Kokkos::parallel_for("ComputeAuxE-1",
                           range,
                           ComputeAuxE_kernel<Dim2, Metric<Dim2>>(mblock.em0,
                                                                  mblock.em,
                                                                  mblock.aux,
                                                                  mblock.metric));
    } else if (g == gr_getE::D_B0) {
      Kokkos::parallel_for("ComputeAuxE-2",
                           range,
                           ComputeAuxE_kernel<Dim2, Metric<Dim2>>(mblock.em,
                                                                  mblock.em0,
                                                                  mblock.aux,
                                                                  mblock.metric));
    } else {
      NTTHostError("Wrong option for `g`");
    }
    NTTLog();
  }

  template <>
  void GRPIC<Dim2>::ComputeAuxH(const gr_getH& g) {
    auto& mblock = this->meshblock;
    auto range = CreateRangePolicy<Dim2>({ mblock.i1_min() - 1, mblock.i2_min() },
                                         { mblock.i1_max(), mblock.i2_max() + 1 });
    if (g == gr_getH::D_B0) {
      Kokkos::parallel_for("ComputeAuxH-1",
                           range,
                           ComputeAuxH_kernel<Dim2, Metric<Dim2>>(mblock.em,
                                                                  mblock.em0,
                                                                  mblock.aux,
                                                                  mblock.metric));
    } else if (g == gr_getH::D0_B0) {
      Kokkos::parallel_for("ComputeAuxH-2",
                           range,
                           ComputeAuxH_kernel<Dim2, Metric<Dim2>>(mblock.em0,
                                                                  mblock.em0,
                                                                  mblock.aux,
                                                                  mblock.metric));
    } else {
      NTTHostError("Wrong option for `g`");
    }
    NTTLog();
  }

  template <>
  void GRPIC<Dim2>::TimeAverageDB() {
    auto& mblock = this->meshblock;
    auto  DB0    = mblock.em0;
    auto  DB     = mblock.em;
    Kokkos::parallel_for(
      "TimeAverageDB",
      mblock.rangeActiveCells(),
      Lambda(index_t i1, index_t i2) {
        DB0(i1, i2, em::bx1) = HALF * (DB0(i1, i2, em::bx1) + DB(i1, i2, em::bx1));
        DB0(i1, i2, em::bx2) = HALF * (DB0(i1, i2, em::bx2) + DB(i1, i2, em::bx2));
        DB0(i1, i2, em::bx3) = HALF * (DB0(i1, i2, em::bx3) + DB(i1, i2, em::bx3));
        DB0(i1, i2, em::dx1) = HALF * (DB0(i1, i2, em::dx1) + DB(i1, i2, em::dx1));
        DB0(i1, i2, em::dx2) = HALF * (DB0(i1, i2, em::dx2) + DB(i1, i2, em::dx2));
        DB0(i1, i2, em::dx3) = HALF * (DB0(i1, i2, em::dx3) + DB(i1, i2, em::dx3));
      });
    NTTLog();
  }

  template <>
  void GRPIC<Dim2>::TimeAverageJ() {
    auto& mblock = this->meshblock;
    auto  J0     = mblock.cur0;
    auto  J      = mblock.cur;
    Kokkos::parallel_for(
      "TimeAverageJ",
      mblock.rangeActiveCells(),
      Lambda(index_t i1, index_t i2) {
        J(i1, i2, cur::jx1) = HALF * (J0(i1, i2, cur::jx1) + J(i1, i2, cur::jx1));
        J(i1, i2, cur::jx2) = HALF * (J0(i1, i2, cur::jx2) + J(i1, i2, cur::jx2));
        J(i1, i2, cur::jx3) = HALF * (J0(i1, i2, cur::jx3) + J(i1, i2, cur::jx3));
      });
    NTTLog();
  }

  template <>
  void GRPIC<Dim3>::ComputeAuxE(const gr_getE&) {
    NTTHostError("not implemented");
  }

  template <>
  void GRPIC<Dim3>::ComputeAuxH(const gr_getH&) {
    NTTHostError("not implemented");
  }

  template <>
  void GRPIC<Dim3>::TimeAverageDB() {
    NTTHostError("not implemented");
  }

  template <>
  void GRPIC<Dim3>::TimeAverageJ() {
    NTTHostError("not implemented");
  }

} // namespace ntt