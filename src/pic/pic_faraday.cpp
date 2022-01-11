#include "global.h"
#include "pic.h"

#include "pic_faraday_minkowski.hpp"
#include "pic_faraday_curvilinear.hpp"

#include <stdexcept>

namespace ntt {
  template <>
  void PIC<Dimension::ONE_D>::faradaySubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    if (m_mblock.metric->label == "minkowski") {
      // dx is passed only in minkowski case to avoid trivial metric computations.
      const auto dx {(m_mblock.metric->x1_max - m_mblock.metric->x1_min) / m_mblock.metric->nx1};
      Kokkos::parallel_for(
        "faraday", m_mblock.loopActiveCells(), FaradayMinkowski<Dimension::ONE_D>(m_mblock, coeff / dx));
    } else {
      NTTError("faraday for this metric not defined");
    }
  }

  template <>
  void PIC<Dimension::TWO_D>::faradaySubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    if (m_mblock.metric->label == "minkowski") {
      // dx is passed only in minkowski case to avoid trivial metric computations.
      const auto dx {(m_mblock.metric->x1_max - m_mblock.metric->x1_min) / m_mblock.metric->nx1};
      Kokkos::parallel_for(
        "faraday", m_mblock.loopActiveCells(), FaradayMinkowski<Dimension::TWO_D>(m_mblock, coeff / dx));
    } else if ((m_mblock.metric->label == "spherical") || (m_mblock.metric->label == "qspherical")) {
      Kokkos::parallel_for(
        "faraday", m_mblock.loopActiveCells(), FaradayCurvilinear<Dimension::TWO_D>(m_mblock, coeff));
    } else {
      NTTError("faraday for this metric not defined");
    }
  }

  template <>
  void PIC<Dimension::THREE_D>::faradaySubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    if (m_mblock.metric->label == "minkowski") {
      // dx is passed only in minkowski case to avoid trivial metric computations.
      const auto dx {(m_mblock.metric->x1_max - m_mblock.metric->x1_min) / m_mblock.metric->nx1};
      Kokkos::parallel_for(
        "faraday", m_mblock.loopActiveCells(), FaradayMinkowski<Dimension::THREE_D>(m_mblock, coeff / dx));
    } else {
      NTTError("faraday for this metric not defined");
    }
  }

} // namespace ntt