#include "aux.hpp"

#include "wrapper.h"

#include "grpic.h"

#include <stdexcept>

namespace ntt {

  template <>
  void GRPIC<Dim2>::ComputeAuxE(const gr_getE& g) {
    auto& mblock = this->meshblock;
    auto  range  = CreateRangePolicy<Dim2>({ mblock.i1_min() - 1, mblock.i2_min() },
                                         { mblock.i1_max(), mblock.i2_max() + 1 });
    if (g == gr_getE::D0_B) {
      Kokkos::parallel_for("ComputeAuxE-1", range, ComputeAuxE_D0_B_kernel<Dim2>(mblock));
    } else if (g == gr_getE::D_B0) {
      Kokkos::parallel_for("ComputeAuxE-2", range, ComputeAuxE_D_B0_kernel<Dim2>(mblock));
    } else {
      NTTHostError("Wrong option for `g`");
    }
  }

  template <>
  void GRPIC<Dim2>::ComputeAuxH(const gr_getH& g) {
    auto& mblock = this->meshblock;
    auto  range  = CreateRangePolicy<Dim2>({ mblock.i1_min() - 1, mblock.i2_min() },
                                         { mblock.i1_max(), mblock.i2_max() + 1 });
    if (g == gr_getH::D_B0) {
      Kokkos::parallel_for("ComputeAuxH-1", range, ComputeAuxH_D_B0_kernel<Dim2>(mblock));
    } else if (g == gr_getH::D0_B0) {
      Kokkos::parallel_for("ComputeAuxH-2", range, ComputeAuxH_D0_B0_kernel<Dim2>(mblock));
    } else {
      NTTHostError("Wrong option for `g`");
    }
  }

  template <>
  void GRPIC<Dim2>::TimeAverageDB() {
    auto& mblock = this->meshblock;
    Kokkos::parallel_for(
      "TimeAverageDB", mblock.rangeActiveCells(), TimeAverageDB_kernel<Dim2>(mblock));
  }

  template <>
  void GRPIC<Dim2>::TimeAverageJ() {
    auto& mblock = this->meshblock;
    Kokkos::parallel_for(
      "TimeAverageJ", mblock.rangeActiveCells(), TimeAverageJ_kernel<Dim2>(mblock));
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

}    // namespace ntt