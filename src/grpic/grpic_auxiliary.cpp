#include "global.h"
#include "grpic.h"
#include "grpic_auxiliary.hpp"

#include <stdexcept>

namespace ntt {
  const auto Dim2 = Dim2;
  const auto Dim3 = Dim3;

  template <>
  void GRPIC<Dim2>::computeAuxESubstep(const real_t&, const gr_getE& f) {
    auto range {CreateRangePolicy<Dim2>({m_mblock.i1_min() - 1, m_mblock.i2_min()},
                               {m_mblock.i1_max(), m_mblock.i2_max() + 1})};
    if (f == gr_getE::D0_B) {
      Kokkos::parallel_for("auxiliary_E", range, computeAuxE_D0_B<Dim2>(m_mblock));
    } else if (f == gr_getE::D_B0) {
      Kokkos::parallel_for("auxiliary_E", range, computeAuxE_D_B0<Dim2>(m_mblock));
    } else {
      NTTError("Wrong option for `f`");
    }
  }

  template <>
  void GRPIC<Dim2>::computeAuxHSubstep(const real_t&, const gr_getH& f) {
    auto range {CreateRangePolicy<Dim2>({m_mblock.i1_min() - 1, m_mblock.i2_min()},
                               {m_mblock.i1_max(), m_mblock.i2_max() + 1})};
    if (f == gr_getH::D_B0) {
      Kokkos::parallel_for("auxiliary_H", range, computeAuxH_D_B0<Dim2>(m_mblock));
    } else if (f == gr_getH::D0_B0) {
      Kokkos::parallel_for("auxiliary_H", range, computeAuxH_D0_B0<Dim2>(m_mblock));
    } else {
      NTTError("Wrong option for `f`");
    }
  }

  template <>
  void GRPIC<Dim2>::timeAverageDBSubstep(const real_t&) {
    Kokkos::parallel_for(
      "auxiliary_EM", m_mblock.rangeActiveCells(), timeAverageDB<Dim2>(m_mblock));
  }

  template <>
  void GRPIC<Dim2>::timeAverageJSubstep(const real_t&) {
    Kokkos::parallel_for(
      "auxiliary_J", m_mblock.rangeActiveCells(), timeAverageJ<Dim2>(m_mblock));
  }

  template <>
  void GRPIC<Dim3>::computeAuxESubstep(const real_t&, const gr_getE&) {
    NTTError("3D GRPIC not implemented yet");
  }

  template <>
  void GRPIC<Dim3>::computeAuxHSubstep(const real_t&, const gr_getH&) {
    NTTError("3D GRPIC not implemented yet");
  }

  template <>
  void GRPIC<Dim3>::timeAverageDBSubstep(const real_t&) {
    NTTError("3D GRPIC not implemented yet");
  }

  template <>
  void GRPIC<Dim3>::timeAverageJSubstep(const real_t&) {
    NTTError("3D GRPIC not implemented yet");
  }

} // namespace ntt