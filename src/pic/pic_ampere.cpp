#include "global.h"
#include "pic.h"

#if METRIC == MINKOWSKI_METRIC
#  include "pic_ampere_minkowski.hpp"
#else
#  include "pic_ampere_curvilinear.hpp"
#endif

#include <stdexcept>

namespace ntt {
  template <>
  void PIC<Dimension::ONE_D>::ampereSubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
#if METRIC == MINKOWSKI_METRIC
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(m_mblock.metric.x1_max - m_mblock.metric.x1_min) / m_mblock.metric.nx1};
    Kokkos::parallel_for("ampere", m_mblock.loopActiveCells(), AmpereMinkowski<Dimension::ONE_D>(m_mblock, coeff / dx));
#else
    (void)(fraction);
    (void)(coeff);
    NTTError("ampere for this metric not defined");
#endif
  }

  template <>
  void PIC<Dimension::TWO_D>::ampereSubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
#if METRIC == MINKOWSKI_METRIC
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(m_mblock.metric.x1_max - m_mblock.metric.x1_min) / m_mblock.metric.nx1};
    Kokkos::parallel_for("ampere", m_mblock.loopActiveCells(), AmpereMinkowski<Dimension::TWO_D>(m_mblock, coeff / dx));
#else
    Kokkos::parallel_for(
      "ampere",
      NTTRange<Dimension::TWO_D>({m_mblock.i_min(), m_mblock.j_min() + 1}, {m_mblock.i_max(), m_mblock.j_max()}),
      AmpereCurvilinear<Dimension::TWO_D>(m_mblock, coeff));
    Kokkos::parallel_for("ampere_pole",
                         NTTRange<Dimension::ONE_D>({m_mblock.i_min()}, {m_mblock.i_max()}),
                         AmpereCurvilinearPoles<Dimension::TWO_D>(m_mblock, coeff));
#endif
  }

  template <>
  void PIC<Dimension::THREE_D>::ampereSubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
#if METRIC == MINKOWSKI_METRIC
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(m_mblock.metric.x1_max - m_mblock.metric.x1_min) / m_mblock.metric.nx1};
    Kokkos::parallel_for(
      "ampere", m_mblock.loopActiveCells(), AmpereMinkowski<Dimension::THREE_D>(m_mblock, coeff / dx));
#else
    (void)(fraction);
    (void)(coeff);
    NTTError("ampere for this metric not defined");
#endif
  }

} // namespace ntt
