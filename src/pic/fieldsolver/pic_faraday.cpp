#include "global.h"
#include "pic.h"

#ifdef MINKOWSKI_METRIC
#  include "pic_faraday_minkowski.hpp"
#else
#  include "pic_faraday_curvilinear.hpp"
#endif

#include <stdexcept>

namespace ntt {
  template <>
  void PIC<Dim1>::faradaySubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
#ifdef MINKOWSKI_METRIC
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(m_mblock.metric.x1_max - m_mblock.metric.x1_min) / m_mblock.metric.nx1};
    Kokkos::parallel_for(
      "faraday", m_mblock.rangeActiveCells(), FaradayMinkowski<Dim1>(m_mblock, coeff / dx));
#else
    (void)(fraction);
    (void)(coeff);
    NTTError("faraday for this metric not defined");
#endif
  }

  template <>
  void PIC<Dim2>::faradaySubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
#ifdef MINKOWSKI_METRIC
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(m_mblock.metric.x1_max - m_mblock.metric.x1_min) / m_mblock.metric.nx1};
    Kokkos::parallel_for(
      "faraday", m_mblock.rangeActiveCells(), FaradayMinkowski<Dim2>(m_mblock, coeff / dx));
#else
    Kokkos::parallel_for(
      "faraday", m_mblock.rangeActiveCells(), FaradayCurvilinear<Dim2>(m_mblock, coeff));
#endif
  }

  template <>
  void PIC<Dim3>::faradaySubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
#ifdef MINKOWSKI_METRIC
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(m_mblock.metric.x1_max - m_mblock.metric.x1_min) / m_mblock.metric.nx1};
    Kokkos::parallel_for(
      "faraday", m_mblock.rangeActiveCells(), FaradayMinkowski<Dim3>(m_mblock, coeff / dx));
#else
    (void)(fraction);
    (void)(coeff);
    NTTError("faraday for this metric not defined");
#endif
  }

} // namespace ntt