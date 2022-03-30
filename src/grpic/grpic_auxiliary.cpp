#include "global.h"
#include "grpic.h"
#include "grpic_auxiliary.hpp"

#include <stdexcept>

namespace ntt {

  template <>
  void GRPIC<Dimension::TWO_D>::computeAuxESubstep(const real_t&, const gr_getE& f) {
    auto range {
      NTTRange<Dimension::TWO_D>({m_mblock.i_min() - 1, m_mblock.j_min()}, {m_mblock.i_max(), m_mblock.j_max() + 1})};
    if (f == gr_getE::D0_B) {
      Kokkos::parallel_for("auxiliary_E", range, computeAuxE_D0_B<Dimension::TWO_D>(m_mblock));
    } else if (f == gr_getE::D_B0) {
      Kokkos::parallel_for("auxiliary_E", range, computeAuxE_D_B0<Dimension::TWO_D>(m_mblock));
    } else {
      NTTError("Wrong option for `f`");
    }
  }

  template <>
  void GRPIC<Dimension::TWO_D>::computeAuxHSubstep(const real_t&, const gr_getH& f) {
    auto range {
      // @CHECK:
      NTTRange<Dimension::TWO_D>({m_mblock.i_min() - 1, m_mblock.j_min()}, {m_mblock.i_max(), m_mblock.j_max() + 1})
      // NTTRange<Dimension::TWO_D>({m_mblock.i_min() - 1, m_mblock.j_min()}, {m_mblock.i_max(), m_mblock.j_max()})};
    };
    if (f == gr_getH::D_B0) {
      Kokkos::parallel_for("auxiliary_H", range, computeAuxH_D_B0<Dimension::TWO_D>(m_mblock));
    } else if (f == gr_getH::D0_B0) {
      Kokkos::parallel_for("auxiliary_H", range, computeAuxH_D0_B0<Dimension::TWO_D>(m_mblock));
    } else {
      NTTError("Wrong option for `f`");
    }
  }

  template <>
  void GRPIC<Dimension::TWO_D>::timeAverageDBSubstep(const real_t&) {
    Kokkos::parallel_for("auxiliary_EM", m_mblock.loopActiveCells(), timeAverageDB<Dimension::TWO_D>(m_mblock));
  }

  template <>
  void GRPIC<Dimension::TWO_D>::timeAverageJSubstep(const real_t&) {
    Kokkos::parallel_for("auxiliary_J", m_mblock.loopActiveCells(), timeAverageJ<Dimension::TWO_D>(m_mblock));
  }

  template <>
  void GRPIC<Dimension::THREE_D>::computeAuxESubstep(const real_t&, const gr_getE&) {
    NTTError("3D GRPIC not implemented yet");
  }

  template <>
  void GRPIC<Dimension::THREE_D>::computeAuxHSubstep(const real_t&, const gr_getH&) {
    NTTError("3D GRPIC not implemented yet");
  }

  template <>
  void GRPIC<Dimension::THREE_D>::timeAverageDBSubstep(const real_t&) {
    NTTError("3D GRPIC not implemented yet");
  }

  template <>
  void GRPIC<Dimension::THREE_D>::timeAverageJSubstep(const real_t&) {
    NTTError("3D GRPIC not implemented yet");
  }

} // namespace ntt