#include "global.h"
#include "pic.h"

#if (METRIC == MINKOWSKI_METRIC)
#  include "pic_ampere_minkowski.hpp"
#else
#  include "pic_ampere_curvilinear.hpp"
#endif

#include <stdexcept>

namespace ntt {
  template <>
  void PIC<Dim1>::ampereSubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
#if (METRIC == MINKOWSKI_METRIC)
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(m_mblock.metric.x1_max - m_mblock.metric.x1_min) / m_mblock.metric.nx1};
    Kokkos::parallel_for(
      "ampere", m_mblock.rangeActiveCells(), AmpereMinkowski<Dim1>(m_mblock, coeff / dx));
#else
    (void)(fraction);
    (void)(coeff);
    NTTError("ampere for this metric not defined");
#endif
  }

  template <>
  void PIC<Dim2>::ampereSubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
#if (METRIC == MINKOWSKI_METRIC)
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(m_mblock.metric.x1_max - m_mblock.metric.x1_min) / m_mblock.metric.nx1};
    Kokkos::parallel_for(
      "ampere", m_mblock.rangeActiveCells(), AmpereMinkowski<Dim2>(m_mblock, coeff / dx));
#else
    Kokkos::parallel_for("ampere",
                         CreateRangePolicy<Dim2>({m_mblock.i1_min(), m_mblock.i2_min() + 1},
                                        {m_mblock.i1_max(), m_mblock.i2_max()}),
                         AmpereCurvilinear<Dim2>(m_mblock, coeff));
    Kokkos::parallel_for("ampere_pole",
                         CreateRangePolicy<Dim1>({m_mblock.i1_min()}, {m_mblock.i1_max()}),
                         AmpereCurvilinearPoles<Dim2>(m_mblock, coeff));
#endif
  }

  template <>
  void PIC<Dim3>::ampereSubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
#if (METRIC == MINKOWSKI_METRIC)
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(m_mblock.metric.x1_max - m_mblock.metric.x1_min) / m_mblock.metric.nx1};
    Kokkos::parallel_for(
      "ampere", m_mblock.rangeActiveCells(), AmpereMinkowski<Dim3>(m_mblock, coeff / dx));
#else
    (void)(fraction);
    (void)(coeff);
    NTTError("ampere for this metric not defined");
#endif
  }

} // namespace ntt