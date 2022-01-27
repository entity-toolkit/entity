#ifdef GRPIC_SIMTYPE

#  include "global.h"
#  include "grpic.h"
#  include "grpic_auxiliary.hpp"

#  include <stdexcept>

namespace ntt {

  template <>
  void GRPIC<Dimension::TWO_D>::Compute_E_Substep(const real_t&) {
    Kokkos::parallel_for("auxiliary_E", m_mblock.loopActiveCells(), Compute_E<Dimension::TWO_D>(m_mblock));
  }

  template <>
  void GRPIC<Dimension::TWO_D>::Compute_H_Substep(const real_t&) {
    Kokkos::parallel_for("auxiliary_H", m_mblock.loopActiveCells(), Compute_H<Dimension::TWO_D>(m_mblock));
  }

  template <>
  void GRPIC<Dimension::TWO_D>::Average_EM_Substep(const real_t&) {
    Kokkos::parallel_for("auxiliary_EM", m_mblock.loopAllCells(), Average_EM<Dimension::TWO_D>(m_mblock));
  }

  template <>
  void GRPIC<Dimension::TWO_D>::Average_J_Substep(const real_t&) {
    Kokkos::parallel_for("auxiliary_J", m_mblock.loopAllCells(), Average_J<Dimension::TWO_D>(m_mblock));
  }

  template <>
  void GRPIC<Dimension::THREE_D>::Compute_E_Substep(const real_t&) {
    NTTError("auxiliary for this metric not defined");
  }

  template <>
  void GRPIC<Dimension::THREE_D>::Compute_H_Substep(const real_t&) {
    NTTError("auxiliary for this metric not defined");
  }

  template <>
  void GRPIC<Dimension::THREE_D>::Average_EM_Substep(const real_t&) {
    NTTError("auxiliary for this metric not defined");
  }

  template <>
  void GRPIC<Dimension::THREE_D>::Average_J_Substep(const real_t&) {
    NTTError("auxiliary for this metric not defined");
  }

} // namespace ntt

#endif