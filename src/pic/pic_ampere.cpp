#include "global.h"
#include "pic.h"

#include "pic_ampere_minkowski.hpp"

namespace ntt {
  template <>
  void PIC<Dimension::ONE_D>::ampereSubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    if (m_mblock.metric->label == "minkowski") {
      // dx is passed only in minkowski case to avoid trivial metric computations.
      const auto dx {(m_mblock.metric->x1_max - m_mblock.metric->x1_min) / m_mblock.metric->nx1};
      Kokkos::parallel_for(
        "ampere", m_mblock.loopActiveCells(), AmpereMinkowski<Dimension::ONE_D>(m_mblock, coeff / dx));
    } else {
      NTTError("ampere for this metric not defined");
    }
  }

  template <>
  void PIC<Dimension::TWO_D>::ampereSubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    if (m_mblock.metric->label == "minkowski") {
      // dx is passed only in minkowski case to avoid trivial metric computations.
      const auto dx {(m_mblock.metric->x1_max - m_mblock.metric->x1_min) / m_mblock.metric->nx1};
      Kokkos::parallel_for(
        "ampere", m_mblock.loopActiveCells(), AmpereMinkowski<Dimension::TWO_D>(m_mblock, coeff / dx));
    } else {
      NTTError("ampere for this metric not defined");
    }
  }

  template <>
  void PIC<Dimension::THREE_D>::ampereSubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    if (m_mblock.metric->label == "minkowski") {
      // dx is passed only in minkowski case to avoid trivial metric computations.
      const auto dx {(m_mblock.metric->x1_max - m_mblock.metric->x1_min) / m_mblock.metric->nx1};
      Kokkos::parallel_for(
        "ampere", m_mblock.loopActiveCells(), AmpereMinkowski<Dimension::THREE_D>(m_mblock, coeff / dx));
    } else {
      NTTError("ampere for this metric not defined");
    }
  }

} // namespace ntt
