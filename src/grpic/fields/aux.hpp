#ifndef GRPIC_AUXILIARY_H
#define GRPIC_AUXILIARY_H

#include "wrapper.h"

#include "field_macros.h"
#include "io/output.h"
#include "grpic.h"
#include "meshblock/meshblock.h"

namespace ntt {
  /**
   * @brief Kernel for computing E.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class ComputeAuxE_D0_B_kernel {
    Meshblock<D, GRPICEngine> m_mblock;

  public:
    ComputeAuxE_D0_B_kernel(const Meshblock<D, GRPICEngine>& mblock) : m_mblock(mblock) {}
    Inline void operator()(index_t, index_t) const;
  };

  // First calculation, with B and D0
  template <>
  Inline void ComputeAuxE_D0_B_kernel<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

    real_t h_11_iPj { m_mblock.metric.h_11({ i_ + HALF, j_ }) };
    real_t h_22_ijP { m_mblock.metric.h_22({ i_, j_ + HALF }) };
    real_t h_33_ij { m_mblock.metric.h_33({ i_, j_ }) };
    real_t alpha_ij { m_mblock.metric.alpha({ i_, j_ }) };
    real_t alpha_iPj { m_mblock.metric.alpha({ i_ + HALF, j_ }) };
    real_t alpha_ijP { m_mblock.metric.alpha({ i_, j_ + HALF }) };

    real_t w1, w2;
    real_t h_13_ij1, h_13_ij2;
    real_t sqrt_detH_ij1, sqrt_detH_ij2;
    real_t beta_ij1, beta_ij2;

    w1       = m_mblock.metric.sqrt_det_h_tilde({ i_ - HALF, j_ });
    w2       = m_mblock.metric.sqrt_det_h_tilde({ i_ + HALF, j_ });
    h_13_ij1 = m_mblock.metric.h_13({ i_ - HALF, j_ });
    h_13_ij2 = m_mblock.metric.h_13({ i_ + HALF, j_ });
    real_t D1_half { (w1 * h_13_ij1 * D0X1(i - 1, j) + w2 * h_13_ij2 * D0X1(i, j))
                     / (w1 + w2) };

    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({ i_ - HALF, j_ });
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({ i_ + HALF, j_ });
    beta_ij1      = m_mblock.metric.beta1({ i_ - HALF, j_ });
    beta_ij2      = m_mblock.metric.beta1({ i_ + HALF, j_ });
    real_t B2_half { (w1 * sqrt_detH_ij1 * beta_ij1 * BX2(i - 1, j)
                      + w2 * sqrt_detH_ij2 * beta_ij2 * BX2(i, j))
                     / (w1 + w2) };

    w1            = m_mblock.metric.sqrt_det_h_tilde({ i_ - HALF, j_ + HALF });
    w2            = m_mblock.metric.sqrt_det_h_tilde({ i_ + HALF, j_ + HALF });
    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({ i_ - HALF, j_ + HALF });
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({ i_ + HALF, j_ + HALF });
    beta_ij1      = m_mblock.metric.beta1({ i_ - HALF, j_ + HALF });
    beta_ij2      = m_mblock.metric.beta1({ i_ + HALF, j_ + HALF });
    real_t B3_half { (w1 * sqrt_detH_ij1 * beta_ij1 * BX3(i - 1, j)
                      + w2 * sqrt_detH_ij2 * beta_ij2 * BX3(i, j))
                     / (w1 + w2) };

    w1       = m_mblock.metric.sqrt_det_h_tilde({ i_, j_ });
    w2       = m_mblock.metric.sqrt_det_h_tilde({ i_ + ONE, j_ });
    h_13_ij1 = m_mblock.metric.h_13({ i_, j_ });
    h_13_ij2 = m_mblock.metric.h_13({ i_ + ONE, j_ });
    real_t D3_half { (w1 * h_13_ij1 * D0X3(i, j) + w2 * h_13_ij2 * D0X3(i + 1, j))
                     / (w1 + w2) };

    real_t D1_cov { h_11_iPj * D0X1(i, j) + D3_half };
    real_t D2_cov { h_22_ijP * D0X2(i, j) };
    real_t D3_cov { h_33_ij * D0X3(i, j) + D1_half };

    EX1(i, j) = alpha_iPj * D1_cov;
    EX2(i, j) = alpha_ijP * D2_cov - B3_half;
    EX3(i, j) = alpha_ij * D3_cov + B2_half;
  }

  template <Dimension D>
  class ComputeAuxE_D_B0_kernel {
    Meshblock<D, GRPICEngine> m_mblock;

  public:
    ComputeAuxE_D_B0_kernel(const Meshblock<D, GRPICEngine>& mblock) : m_mblock(mblock) {}
    Inline void operator()(index_t, index_t) const;
  };

  // Second calculation, with B0 and D
  template <>
  Inline void ComputeAuxE_D_B0_kernel<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

    real_t h_11_iPj { m_mblock.metric.h_11({ i_ + HALF, j_ }) };
    real_t h_22_ijP { m_mblock.metric.h_22({ i_, j_ + HALF }) };
    real_t h_33_ij { m_mblock.metric.h_33({ i_, j_ }) };
    real_t alpha_ij { m_mblock.metric.alpha({ i_, j_ }) };
    real_t alpha_iPj { m_mblock.metric.alpha({ i_ + HALF, j_ }) };
    real_t alpha_ijP { m_mblock.metric.alpha({ i_, j_ + HALF }) };

    real_t w1, w2;
    real_t h_13_ij1, h_13_ij2;
    real_t sqrt_detH_ij1, sqrt_detH_ij2;
    real_t beta_ij1, beta_ij2;

    w1       = m_mblock.metric.sqrt_det_h_tilde({ i_ - HALF, j_ });
    w2       = m_mblock.metric.sqrt_det_h_tilde({ i_ + HALF, j_ });
    h_13_ij1 = m_mblock.metric.h_13({ i_ - HALF, j_ });
    h_13_ij2 = m_mblock.metric.h_13({ i_ + HALF, j_ });
    real_t D1_half { (w1 * h_13_ij1 * DX1(i - 1, j) + w2 * h_13_ij2 * DX1(i, j)) / (w1 + w2) };

    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({ i_ - HALF, j_ });
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({ i_ + HALF, j_ });
    beta_ij1      = m_mblock.metric.beta1({ i_ - HALF, j_ });
    beta_ij2      = m_mblock.metric.beta1({ i_ + HALF, j_ });
    real_t B2_half { (w1 * sqrt_detH_ij1 * beta_ij1 * B0X2(i - 1, j)
                      + w2 * sqrt_detH_ij2 * beta_ij2 * B0X2(i, j))
                     / (w1 + w2) };

    w1            = m_mblock.metric.sqrt_det_h_tilde({ i_ - HALF, j_ + HALF });
    w2            = m_mblock.metric.sqrt_det_h_tilde({ i_ + HALF, j_ + HALF });
    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({ i_ - HALF, j_ + HALF });
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({ i_ + HALF, j_ + HALF });
    beta_ij1      = m_mblock.metric.beta1({ i_ - HALF, j_ + HALF });
    beta_ij2      = m_mblock.metric.beta1({ i_ + HALF, j_ + HALF });
    real_t B3_half { (w1 * sqrt_detH_ij1 * beta_ij1 * B0X3(i - 1, j)
                      + w2 * sqrt_detH_ij2 * beta_ij2 * B0X3(i, j))
                     / (w1 + w2) };

    h_13_ij1 = m_mblock.metric.h_13({ i_, j_ });
    h_13_ij2 = m_mblock.metric.h_13({ i_ + ONE, j_ });
    w1       = m_mblock.metric.sqrt_det_h_tilde({ i_, j_ });
    w2       = m_mblock.metric.sqrt_det_h_tilde({ i_ + ONE, j_ });
    real_t D3_half { (w1 * h_13_ij1 * DX3(i, j) + w2 * h_13_ij2 * DX3(i + 1, j)) / (w1 + w2) };

    real_t D1_cov { h_11_iPj * DX1(i, j) + D3_half };
    real_t D2_cov { h_22_ijP * DX2(i, j) };
    real_t D3_cov { h_33_ij * DX3(i, j) + D1_half };

    EX1(i, j) = alpha_iPj * D1_cov;
    EX2(i, j) = alpha_ijP * D2_cov - B3_half;
    EX3(i, j) = alpha_ij * D3_cov + B2_half;
  }

  /**
   * @brief Kernel for computing H.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class ComputeAuxH_D_B0_kernel {
    Meshblock<D, GRPICEngine> m_mblock;

  public:
    ComputeAuxH_D_B0_kernel(const Meshblock<D, GRPICEngine>& mblock) : m_mblock(mblock) {}
    Inline void operator()(index_t, index_t) const;
  };

  // First calculation, with B0 and D
  template <>
  Inline void ComputeAuxH_D_B0_kernel<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

    real_t h_11_ijP { m_mblock.metric.h_11({ i_, j_ + HALF }) };
    real_t h_22_iPj { m_mblock.metric.h_22({ i_ + HALF, j_ }) };
    real_t h_33_iPjP { m_mblock.metric.h_33({ i_ + HALF, j_ + HALF }) };
    real_t alpha_ijP { m_mblock.metric.alpha({ i_, j_ + HALF }) };
    real_t alpha_iPj { m_mblock.metric.alpha({ i_ + HALF, j_ }) };
    real_t alpha_iPjP { m_mblock.metric.alpha({ i_ + HALF, j_ + HALF }) };

    real_t w1, w2;
    real_t h_13_ij1, h_13_ij2;
    real_t sqrt_detH_ij1, sqrt_detH_ij2;
    real_t beta_ij1, beta_ij2;

    w1       = m_mblock.metric.sqrt_det_h_tilde({ i_, j_ + HALF });
    w2       = m_mblock.metric.sqrt_det_h_tilde({ i_ + ONE, j_ + HALF });
    h_13_ij1 = m_mblock.metric.h_13({ i_, j_ + HALF });
    h_13_ij2 = m_mblock.metric.h_13({ i_ + ONE, j_ + HALF });
    real_t B1_half { (w1 * h_13_ij1 * B0X1(i, j) + w2 * h_13_ij2 * B0X1(i + 1, j))
                     / (w1 + w2) };

    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({ i_, j_ + HALF });
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({ i_ + ONE, j_ + HALF });
    beta_ij1      = m_mblock.metric.beta1({ i_, j_ + HALF });
    beta_ij2      = m_mblock.metric.beta1({ i_ + ONE, j_ + HALF });
    real_t D2_half { (w1 * sqrt_detH_ij1 * beta_ij1 * DX2(i, j)
                      + w2 * sqrt_detH_ij2 * beta_ij2 * DX2(i + 1, j))
                     / (w1 + w2) };

    w1            = m_mblock.metric.sqrt_det_h_tilde({ i_, j_ });
    w2            = m_mblock.metric.sqrt_det_h_tilde({ i_ + ONE, j_ });
    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({ i_, j_ });
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({ i_ + ONE, j_ });
    beta_ij1      = m_mblock.metric.beta1({ i_, j_ });
    beta_ij2      = m_mblock.metric.beta1({ i_ + ONE, j_ });
    real_t D3_half { (w1 * sqrt_detH_ij1 * beta_ij1 * DX3(i, j)
                      + w2 * sqrt_detH_ij2 * beta_ij2 * DX3(i + 1, j))
                     / (w1 + w2) };

    w1       = m_mblock.metric.sqrt_det_h_tilde({ i_ - HALF, j_ + HALF });
    w2       = m_mblock.metric.sqrt_det_h_tilde({ i_ + HALF, j_ + HALF });
    h_13_ij1 = m_mblock.metric.h_13({ i_ - HALF, j_ + HALF });
    h_13_ij2 = m_mblock.metric.h_13({ i_ + HALF, j_ + HALF });
    real_t B3_half { (w1 * h_13_ij1 * B0X3(i - 1, j) + w2 * h_13_ij2 * B0X3(i, j))
                     / (w1 + w2) };

    real_t B1_cov { h_11_ijP * B0X1(i, j) + B3_half };
    real_t B2_cov { h_22_iPj * B0X2(i, j) };
    real_t B3_cov { h_33_iPjP * B0X3(i, j) + B1_half };

    HX1(i, j) = alpha_ijP * B1_cov;
    HX2(i, j) = alpha_iPj * B2_cov + D3_half;
    HX3(i, j) = alpha_iPjP * B3_cov - D2_half;
  }

  template <Dimension D>
  class ComputeAuxH_D0_B0_kernel {
    Meshblock<D, GRPICEngine> m_mblock;

  public:
    ComputeAuxH_D0_B0_kernel(const Meshblock<D, GRPICEngine>& mblock) : m_mblock(mblock) {}
    Inline void operator()(index_t, index_t) const;
  };

  // Second calculation, with B0 and D0
  template <>
  Inline void ComputeAuxH_D0_B0_kernel<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

    real_t h_11_ijP { m_mblock.metric.h_11({ i_, j_ + HALF }) };
    real_t h_22_iPj { m_mblock.metric.h_22({ i_ + HALF, j_ }) };
    real_t h_33_iPjP { m_mblock.metric.h_33({ i_ + HALF, j_ + HALF }) };
    real_t alpha_ijP { m_mblock.metric.alpha({ i_, j_ + HALF }) };
    real_t alpha_iPj { m_mblock.metric.alpha({ i_ + HALF, j_ }) };
    real_t alpha_iPjP { m_mblock.metric.alpha({ i_ + HALF, j_ + HALF }) };

    real_t w1, w2;
    real_t h_13_ij1, h_13_ij2;
    real_t sqrt_detH_ij1, sqrt_detH_ij2;
    real_t beta_ij1, beta_ij2;

    w1       = m_mblock.metric.sqrt_det_h_tilde({ i_, j_ + HALF });
    w2       = m_mblock.metric.sqrt_det_h_tilde({ i_ + ONE, j_ + HALF });
    h_13_ij1 = m_mblock.metric.h_13({ i_, j_ + HALF });
    h_13_ij2 = m_mblock.metric.h_13({ i_ + ONE, j_ + HALF });
    real_t B1_half { (w1 * h_13_ij1 * B0X1(i, j) + w2 * h_13_ij2 * B0X1(i + 1, j))
                     / (w1 + w2) };

    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({ i_, j_ + HALF });
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({ i_ + ONE, j_ + HALF });
    beta_ij1      = m_mblock.metric.beta1({ i_, j_ + HALF });
    beta_ij2      = m_mblock.metric.beta1({ i_ + ONE, j_ + HALF });
    real_t D2_half { (w1 * sqrt_detH_ij1 * beta_ij1 * D0X2(i, j)
                      + w2 * sqrt_detH_ij2 * beta_ij2 * D0X2(i + 1, j))
                     / (w1 + w2) };

    w1            = m_mblock.metric.sqrt_det_h_tilde({ i_, j_ });
    w2            = m_mblock.metric.sqrt_det_h_tilde({ i_ + ONE, j_ });
    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({ i_, j_ });
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({ i_ + ONE, j_ });
    beta_ij1      = m_mblock.metric.beta1({ i_, j_ });
    beta_ij2      = m_mblock.metric.beta1({ i_ + ONE, j_ });
    real_t D3_half { (w1 * sqrt_detH_ij1 * beta_ij1 * D0X3(i, j)
                      + w2 * sqrt_detH_ij2 * beta_ij2 * D0X3(i + 1, j))
                     / (w1 + w2) };

    w1       = m_mblock.metric.sqrt_det_h_tilde({ i_ - HALF, j_ + HALF });
    w2       = m_mblock.metric.sqrt_det_h_tilde({ i_ + HALF, j_ + HALF });
    h_13_ij1 = m_mblock.metric.h_13({ i_ - HALF, j_ + HALF });
    h_13_ij2 = m_mblock.metric.h_13({ i_ + HALF, j_ + HALF });
    real_t B3_half { (w1 * h_13_ij1 * B0X3(i - 1, j) + w2 * h_13_ij2 * B0X3(i, j))
                     / (w1 + w2) };

    real_t B1_cov { h_11_ijP * B0X1(i, j) + B3_half };
    real_t B2_cov { h_22_iPj * B0X2(i, j) };
    real_t B3_cov { h_33_iPjP * B0X3(i, j) + B1_half };

    HX1(i, j) = alpha_ijP * B1_cov;
    HX2(i, j) = alpha_iPj * B2_cov + D3_half;
    HX3(i, j) = alpha_iPjP * B3_cov - D2_half;
  }

  /**
   * @brief Time average B and D field.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class TimeAverageDB_kernel {
    Meshblock<D, GRPICEngine> m_mblock;

  public:
    TimeAverageDB_kernel(const Meshblock<D, GRPICEngine>& mblock) : m_mblock(mblock) {}
    Inline void operator()(index_t, index_t) const;
  };

  template <>
  Inline void TimeAverageDB_kernel<Dim2>::operator()(index_t i, index_t j) const {
    B0X1(i, j) = HALF * (B0X1(i, j) + BX1(i, j));
    B0X2(i, j) = HALF * (B0X2(i, j) + BX2(i, j));
    B0X3(i, j) = HALF * (B0X3(i, j) + BX3(i, j));
    D0X1(i, j) = HALF * (D0X1(i, j) + DX1(i, j));
    D0X2(i, j) = HALF * (D0X2(i, j) + DX2(i, j));
    D0X3(i, j) = HALF * (D0X3(i, j) + DX3(i, j));
  }

  /**
   * @brief Time average currents.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class TimeAverageJ_kernel {
    Meshblock<D, GRPICEngine> m_mblock;

  public:
    TimeAverageJ_kernel(const Meshblock<D, GRPICEngine>& mblock) : m_mblock(mblock) {}
    Inline void operator()(index_t, index_t) const;
  };

  template <>
  Inline void TimeAverageJ_kernel<Dim2>::operator()(index_t i, index_t j) const {
    JX1(i, j) = HALF * (J0X1(i, j) + JX1(i, j));
    JX2(i, j) = HALF * (J0X2(i, j) + JX2(i, j));
    JX3(i, j) = HALF * (J0X3(i, j) + JX3(i, j));
  }

}    // namespace ntt

#endif
