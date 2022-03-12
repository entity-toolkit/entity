#include "global.h"
#include "grpic.h"
#include "grpic_auxiliary.hpp"

#include <stdexcept>

namespace ntt {

  template <>
  void GRPIC<Dimension::TWO_D>::Compute_E_Substep(const real_t&, const short (&s)) {
    if (s == 0) {
    Kokkos::parallel_for("auxiliary_E", 
    NTTRange<Dimension::TWO_D>({m_mblock.i_min() - 1, m_mblock.j_min()}, {m_mblock.i_max(), m_mblock.j_max() + 1}), 
    Compute_E0<Dimension::TWO_D>(m_mblock));
    } else if (s == 1) {
    Kokkos::parallel_for("auxiliary_E",
    NTTRange<Dimension::TWO_D>({m_mblock.i_min() - 1, m_mblock.j_min()}, {m_mblock.i_max(), m_mblock.j_max() + 1}), 
    Compute_E<Dimension::TWO_D>(m_mblock));
    } else {
    NTTError("Only two options: 0 and 1");
    }
  }

  template <>
  void GRPIC<Dimension::TWO_D>::Compute_H_Substep(const real_t&, const short (&s)) {
    if (s == 0) {
    Kokkos::parallel_for("auxiliary_H", 
    NTTRange<Dimension::TWO_D>({m_mblock.i_min() - 1, m_mblock.j_min()}, {m_mblock.i_max(), m_mblock.j_max() + 1}), 
    Compute_H0<Dimension::TWO_D>(m_mblock));
    } else if (s == 1) {
    Kokkos::parallel_for("auxiliary_H",
    NTTRange<Dimension::TWO_D>({m_mblock.i_min() - 1, m_mblock.j_min()}, {m_mblock.i_max(), m_mblock.j_max() + 1}), 
    Compute_H<Dimension::TWO_D>(m_mblock));
    } else {
    NTTError("Only two options: 0 and 1");
    }
  }

  template <>
  void GRPIC<Dimension::TWO_D>::Average_EM_Substep(const real_t&) {
    Kokkos::parallel_for("auxiliary_EM", m_mblock.loopActiveCells(), Average_EM<Dimension::TWO_D>(m_mblock));
  }

  template <>
  void GRPIC<Dimension::TWO_D>::Average_J_Substep(const real_t&) {
    Kokkos::parallel_for("auxiliary_J", m_mblock.loopActiveCells(), Average_J<Dimension::TWO_D>(m_mblock));
  }

  template <>
  void GRPIC<Dimension::THREE_D>::Compute_E_Substep(const real_t&, const short (&s)) {
    (void)(s);
    NTTError("auxiliary for this metric not defined");
  }

  template <>
  void GRPIC<Dimension::THREE_D>::Compute_H_Substep(const real_t&, const short (&s)) {
    (void)(s);
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