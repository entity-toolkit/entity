#if SIMTYPE == GRPIC_SIMTYPE

#  include "global.h"
#  include "grpic.h"
#  include "grpic_auxiliary.hpp"

#  include <stdexcept>

namespace ntt {

  template <>
  void GRPIC<Dimension::TWO_D>::Compute_E_Substep(const real_t&) {

    auto range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace, Compute_E>(
      {static_cast<range_t>(m_mblock.i_min()), static_cast<range_t>(m_mblock.j_min())},
      {static_cast<range_t>(m_mblock.i_max()), static_cast<range_t>(m_mblock.j_max())});

    Kokkos::parallel_for("auxiliary", range_policy, Compute_auxiliary<Dimension::TWO_D>(m_mblock));
  }

  template <>
  void GRPIC<Dimension::TWO_D>::Compute_H_Substep(const real_t&) {

    auto range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace, Compute_H>(
      {static_cast<range_t>(m_mblock.i_min()), static_cast<range_t>(m_mblock.i_min())},
      {static_cast<range_t>(m_mblock.i_min()), static_cast<range_t>(m_mblock.i_min())});

    Kokkos::parallel_for("auxiliary", range_policy, Compute_auxiliary<Dimension::TWO_D>(m_mblock));
  }

  template <>
  void GRPIC<Dimension::THREE_D>::Compute_E_Substep(const real_t&) {
    NTTError("auxiliary for this metric not defined");
  }

  template <>
  void GRPIC<Dimension::THREE_D>::Compute_H_Substep(const real_t&) {
    NTTError("auxiliary for this metric not defined");
  }

} // namespace ntt

#endif