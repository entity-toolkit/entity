#ifndef GRPIC_FARADAY_H
#define GRPIC_FARADAY_H

#include "global.h"
#include "fields.h"
#include "meshblock.h"
#include "grpic.h"

#include <stdexcept>

namespace ntt {

  /**
   * Algorithms for Faraday's law.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class FaradayGR_aux {
    
    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t                              m_coeff;
    index_t                             j_min;

  public:
    FaradayGR_aux(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock {mblock}, m_coeff {coeff}, j_min {static_cast<index_t>(m_mblock.i2_min())} {}
    Inline void operator()(index_t, index_t) const;
    Inline void operator()(index_t, index_t, index_t) const;
  };

  // First push, updates B0.
  template <>
  Inline void FaradayGR_aux<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};
    // index_t j_min {static_cast<index_t>(m_mblock.i2_min())};

    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    real_t inv_sqrt_detH_iPjP {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_ + HALF})};

    m_mblock.em0(i, j, em::bx1)
      += m_coeff * inv_sqrt_detH_ijP * (m_mblock.aux(i, j, em::ex3) - m_mblock.aux(i, j + 1, em::ex3));

    if (j == j_min) {
      m_mblock.em0(i, j, em::bx2) = ZERO;
    } else {
      m_mblock.em0(i, j, em::bx2)
        += m_coeff * inv_sqrt_detH_iPj * (m_mblock.aux(i + 1, j, em::ex3) - m_mblock.aux(i, j, em::ex3));
    }
    m_mblock.em0(i, j, em::bx3) += m_coeff * inv_sqrt_detH_iPjP
                                   * (m_mblock.aux(i, j + 1, em::ex1) - m_mblock.aux(i, j, em::ex1)
                                      + m_mblock.aux(i, j, em::ex2) - m_mblock.aux(i + 1, j, em::ex2));
  }

  template <>
  Inline void FaradayGR_aux<Dim3>::operator()(index_t, index_t, index_t) const {
    NTTError("3D GRPIC not implemented yet");
  }

  template <Dimension D>
  class FaradayGR {
    
    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t                              m_coeff;
    index_t                             j_min;

  public:
    FaradayGR(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock {mblock}, m_coeff {coeff}, j_min {static_cast<index_t>(m_mblock.i2_min())} {}
    Inline void operator()(index_t, index_t) const;
    Inline void operator()(index_t, index_t, index_t) const;
  };

  // Second push, updates B but assigns it to B0.
  template <>
  Inline void FaradayGR<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};

    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    real_t inv_sqrt_detH_iPjP {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_ + HALF})};

    m_mblock.em0(i, j, em::bx1)
      = m_mblock.em(i, j, em::bx1)
        + m_coeff * inv_sqrt_detH_ijP * (m_mblock.aux(i, j, em::ex3) - m_mblock.aux(i, j + 1, em::ex3));

    if (j == j_min) {
      m_mblock.em0(i, j, em::bx2) = ZERO;
    } else {
      m_mblock.em0(i, j, em::bx2)
        = m_mblock.em(i, j, em::bx2)
          + m_coeff * inv_sqrt_detH_iPj * (m_mblock.aux(i + 1, j, em::ex3) - m_mblock.aux(i, j, em::ex3));
    }

    m_mblock.em0(i, j, em::bx3) = m_mblock.em(i, j, em::bx3)
                                  + m_coeff * inv_sqrt_detH_iPjP
                                      * (m_mblock.aux(i, j + 1, em::ex1) - m_mblock.aux(i, j, em::ex1)
                                         + m_mblock.aux(i, j, em::ex2) - m_mblock.aux(i + 1, j, em::ex2));
  }

  template <>
  Inline void FaradayGR<Dim3>::operator()(index_t, index_t, index_t) const {
    NTTError("3D GRPIC not implemented yet");
  }

} // namespace ntt

#endif