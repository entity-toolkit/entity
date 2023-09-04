#ifndef GRPIC_FARADAY_H
#define GRPIC_FARADAY_H

#include "wrapper.h"

#include "field_macros.h"
#include "grpic.h"

#include "io/output.h"
#include "meshblock/meshblock.h"

namespace ntt {

  /**
   * @brief Algorithms for Faraday's law.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class FaradayAux_kernel {
    Meshblock<D, GRPICEngine> m_mblock;
    real_t                    m_coeff;
    index_t                   j_min;

  public:
    FaradayAux_kernel(const Meshblock<D, GRPICEngine>& mblock, const real_t& coeff) :
      m_mblock { mblock },
      m_coeff { coeff },
      j_min { static_cast<index_t>(m_mblock.i2_min()) } {}

    Inline void operator()(index_t, index_t) const;
  };

  // First push, updates B0.
  template <>
  Inline void FaradayAux_kernel<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

    real_t inv_sqrt_detH_iPj { ONE / m_mblock.metric.sqrt_det_h({ i_ + HALF, j_ }) };
    real_t inv_sqrt_detH_ijP { ONE / m_mblock.metric.sqrt_det_h({ i_, j_ + HALF }) };
    real_t inv_sqrt_detH_iPjP { ONE / m_mblock.metric.sqrt_det_h(
                                        { i_ + HALF, j_ + HALF }) };

    B0X1(i, j) += m_coeff * inv_sqrt_detH_ijP * (EX3(i, j) - EX3(i, j + 1));

    if (j == j_min) {
      B0X2(i, j) = ZERO;
    } else {
      B0X2(i, j) += m_coeff * inv_sqrt_detH_iPj * (EX3(i + 1, j) - EX3(i, j));
    }
    B0X3(i, j) += m_coeff * inv_sqrt_detH_iPjP *
                  (EX1(i, j + 1) - EX1(i, j) + EX2(i, j) - EX2(i + 1, j));
  }

  template <Dimension D>
  class Faraday_kernel {
    Meshblock<D, GRPICEngine> m_mblock;
    real_t                    m_coeff;
    index_t                   j_min;

  public:
    Faraday_kernel(const Meshblock<D, GRPICEngine>& mblock, const real_t& coeff) :
      m_mblock { mblock },
      m_coeff { coeff },
      j_min { static_cast<index_t>(m_mblock.i2_min()) } {}

    Inline void operator()(index_t, index_t) const;
  };

  // Second push, updates B but assigns it to B0.
  template <>
  Inline void Faraday_kernel<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

    real_t inv_sqrt_detH_iPj { ONE / m_mblock.metric.sqrt_det_h({ i_ + HALF, j_ }) };
    real_t inv_sqrt_detH_ijP { ONE / m_mblock.metric.sqrt_det_h({ i_, j_ + HALF }) };
    real_t inv_sqrt_detH_iPjP { ONE / m_mblock.metric.sqrt_det_h(
                                        { i_ + HALF, j_ + HALF }) };

    B0X1(i, j) = BX1(i, j) +
                 m_coeff * inv_sqrt_detH_ijP * (EX3(i, j) - EX3(i, j + 1));

    if (j == j_min) {
      B0X2(i, j) = ZERO;
    } else {
      B0X2(i, j) = BX2(i, j) +
                   m_coeff * inv_sqrt_detH_iPj * (EX3(i + 1, j) - EX3(i, j));
    }

    B0X3(i, j) = BX3(i, j) +
                 m_coeff * inv_sqrt_detH_iPjP *
                   (EX1(i, j + 1) - EX1(i, j) + EX2(i, j) - EX2(i + 1, j));
  }

} // namespace ntt

#endif
