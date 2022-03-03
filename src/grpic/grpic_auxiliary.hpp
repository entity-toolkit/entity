#ifndef GRPIC_AUXILIARY_H
#define GRPIC_AUXILIARY_H

#include "global.h"
#include "fields.h"
#include "meshblock.h"
#include "grpic.h"

#include <stdexcept>

namespace ntt {
  /**
   * Methods for computing E.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Compute_E0 {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;

  public:
    Compute_E0(const Meshblock<D, SimulationType::GRPIC>& mblock)
      : m_mblock(mblock) {}
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  // First calculation, with B and D0
  template <>
  Inline void Compute_E0<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(j - N_GHOSTS)};
    index_t j_min {static_cast<index_t>(m_mblock.j_min())};
    index_t j_max {static_cast<index_t>(m_mblock.j_max())};

    real_t h_11_iPj  {m_mblock.metric.h_11({i_ + HALF, j_})};
    real_t h_13_iPj  {m_mblock.metric.h_13({i_ + HALF, j_})};
    real_t h_22_ijP  {m_mblock.metric.h_22({i_, j_ + HALF})};
    real_t h_33_ij   {m_mblock.metric.h_33({i_, j_})};
    real_t h_13_ij   {m_mblock.metric.h_13({i_, j_})};
    real_t alpha_ij  {m_mblock.metric.alpha({i_, j_})};
    real_t alpha_iPj {m_mblock.metric.alpha({i_ + HALF, j_})};
    real_t alpha_ijP {m_mblock.metric.alpha({i_, j_ + HALF})};
    real_t beta_ij   {m_mblock.metric.beta1u({i_, j_})};
    real_t beta_ijP  {m_mblock.metric.beta1u({i_, j_ + HALF})};

    real_t sqrt_detH_ijP {m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    real_t sqrt_detH_ij {m_mblock.metric.sqrt_det_h({i_, j_})};

    // // B contra interpolation at half cell
    // real_t B2_half {HALF * (m_mblock.em(i - 1, j, em::bx2) + m_mblock.em(i, j, em::bx2))};
    // real_t B3_half {HALF * (m_mblock.em(i - 1, j, em::bx3) + m_mblock.em(i, j, em::bx3))};

    // // D contra interpolation at half cell
    // real_t D1_half {HALF * (m_mblock.em0(i - 1, j, em::ex1) + m_mblock.em0(i, j, em::ex1))};
    // real_t D3_half {HALF * (m_mblock.em0(i, j, em::ex3) + m_mblock.em0(i + 1, j, em::ex3))};

    // // Contravariant D to covariant D
    // real_t D1_cov {h_11_iPj * m_mblock.em0(i, j, em::ex1) + h_13_iPj * D3_half};
    // real_t D2_cov {h_22_ijP * m_mblock.em0(i, j, em::ex2)};
    // real_t D3_cov {h_33_ij  * m_mblock.em0(i, j, em::ex3) + h_13_ij * D1_half};

    // // Compute E_i
    // m_mblock.aux(i, j, em::ex1) = alpha_iPj * D1_cov;
    // m_mblock.aux(i, j, em::ex2) = alpha_ijP * D2_cov - sqrt_detH_ijP * beta_ijP * B3_half;
    // m_mblock.aux(i, j, em::ex3) = alpha_ij  * D3_cov + sqrt_detH_ij  * beta_ij  * B2_half;

    real_t w1, w2;
    real_t h_13_ij1, h_13_ij2; 
    real_t sqrt_detH_ij1, sqrt_detH_ij2;
    real_t beta_ij1, beta_ij2;
    
    w1 = m_mblock.metric.sqrt_det_h_tilde({i_ - HALF, j_});
    w2 = m_mblock.metric.sqrt_det_h_tilde({i_ + HALF, j_});
    h_13_ij1 = m_mblock.metric.h_13({i_ - HALF, j_});
    h_13_ij2 = m_mblock.metric.h_13({i_ + HALF, j_});
    real_t D1_half {(w1 * h_13_ij1 * m_mblock.em0(i - 1, j, em::ex1) 
                   + w2 * h_13_ij2 * m_mblock.em0(i, j, em::ex1)) 
                   / (w1 + w2)};

    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({i_ - HALF, j_});
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({i_ + HALF, j_});
    beta_ij1      = m_mblock.metric.beta1u({i_ - HALF, j_});
    beta_ij2      = m_mblock.metric.beta1u({i_ + HALF, j_});
    real_t B2_half {(w1 * sqrt_detH_ij1 * beta_ij1 * m_mblock.em(i - 1, j, em::bx2) 
                   + w2 * sqrt_detH_ij2 * beta_ij2 * m_mblock.em(i, j, em::bx2)) 
                   / (w1 + w2)};    
    
    w1 = m_mblock.metric.sqrt_det_h_tilde({i_ - HALF, j_ + HALF});
    w2 = m_mblock.metric.sqrt_det_h_tilde({i_ + HALF, j_ + HALF});
    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({i_ - HALF, j_ + HALF});
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({i_ + HALF, j_ + HALF});
    beta_ij1      = m_mblock.metric.beta1u({i_ - HALF, j_ + HALF});
    beta_ij2      = m_mblock.metric.beta1u({i_ + HALF, j_ + HALF});
    real_t B3_half {(w1 * sqrt_detH_ij1 * beta_ij1 * m_mblock.em(i - 1, j, em::bx3) 
                   + w2 * sqrt_detH_ij2 * beta_ij2 * m_mblock.em(i, j, em::bx3)) 
                   / (w1 + w2)};

    w1 = m_mblock.metric.sqrt_det_h_tilde({i_, j_});
    w2 = m_mblock.metric.sqrt_det_h_tilde({i_ + ONE, j_});
    h_13_ij1 = m_mblock.metric.h_13({i_, j_});
    h_13_ij2 = m_mblock.metric.h_13({i_ + ONE, j_});
    real_t D3_half {(w1 * h_13_ij1 * m_mblock.em0(i, j, em::ex3) 
                   + w2 * h_13_ij2 * m_mblock.em0(i + 1, j, em::ex3)) 
                   / (w1 + w2)};

    real_t D1_cov {h_11_iPj * m_mblock.em0(i, j, em::ex1) + D3_half};
    real_t D2_cov {h_22_ijP * m_mblock.em0(i, j, em::ex2)};
    real_t D3_cov {h_33_ij  * m_mblock.em0(i, j, em::ex3) + D1_half};

    m_mblock.aux(i, j, em::ex1) = alpha_iPj * D1_cov;
    m_mblock.aux(i, j, em::ex2) = alpha_ijP * D2_cov - B3_half;
    m_mblock.aux(i, j, em::ex3) = alpha_ij  * D3_cov + B2_half;
   }

  template <>
  Inline void
  Compute_E0<Dimension::THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    // 3d grpic not implemented
  }

  template <Dimension D>
  class Compute_E {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;

  public:
    Compute_E(const Meshblock<D, SimulationType::GRPIC>& mblock)
      : m_mblock(mblock) {}
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  // Second calculation, with B0 and D
  template <>
  Inline void Compute_E<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(j - N_GHOSTS)};
    index_t j_min {static_cast<index_t>(m_mblock.j_min())};
    index_t j_max {static_cast<index_t>(m_mblock.j_max())};
    
    real_t h_11_iPj  {m_mblock.metric.h_11({i_ + HALF, j_})};
    real_t h_13_iPj  {m_mblock.metric.h_13({i_ + HALF, j_})};
    real_t h_22_ijP  {m_mblock.metric.h_22({i_, j_ + HALF})};
    real_t h_33_ij   {m_mblock.metric.h_33({i_, j_})};
    real_t h_13_ij   {m_mblock.metric.h_13({i_, j_})};
    real_t alpha_ij  {m_mblock.metric.alpha({i_, j_})};
    real_t alpha_iPj {m_mblock.metric.alpha({i_ + HALF, j_})};
    real_t alpha_ijP {m_mblock.metric.alpha({i_, j_ + HALF})};
    real_t beta_ij   {m_mblock.metric.beta1u({i_, j_})};
    real_t beta_ijP  {m_mblock.metric.beta1u({i_, j_ + HALF})};
    
    real_t sqrt_detH_ijP {m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    real_t sqrt_detH_ij {m_mblock.metric.sqrt_det_h({i_, j_})};

    // // B contra interpolation at half cell
    // real_t B2_half {HALF * (m_mblock.em0(i - 1, j, em::bx2) + m_mblock.em0(i, j, em::bx2))};
    // real_t B3_half {HALF * (m_mblock.em0(i - 1, j, em::bx3) + m_mblock.em0(i, j, em::bx3))};

    // // D contra interpolation at half cell
    // real_t D1_half {HALF * (m_mblock.em(i - 1, j, em::ex1) + m_mblock.em(i, j, em::ex1))};
    // real_t D3_half {HALF * (m_mblock.em(i, j, em::ex3) + m_mblock.em(i + 1, j, em::ex3))};
 
    // // Contravariant D to covariant D
    // real_t D1_cov {h_11_iPj * m_mblock.em(i, j, em::ex1) + h_13_iPj * D3_half};
    // real_t D2_cov {h_22_ijP * m_mblock.em(i, j, em::ex2)};
    // real_t D3_cov {h_33_ij * m_mblock.em(i, j, em::ex3) + h_13_ij * D1_half};

    // // Compute E_i
    // m_mblock.aux(i, j, em::ex1) = alpha_iPj * D1_cov;
    // m_mblock.aux(i, j, em::ex2) = alpha_ijP * D2_cov - sqrt_detH_ijP * beta_ijP * B3_half;
    // m_mblock.aux(i, j, em::ex3) = alpha_ij  * D3_cov + sqrt_detH_ij  * beta_ij  * B2_half;

    real_t w1, w2;
    real_t h_13_ij1, h_13_ij2; 
    real_t sqrt_detH_ij1, sqrt_detH_ij2;
    real_t beta_ij1, beta_ij2;
    
    w1 = m_mblock.metric.sqrt_det_h_tilde({i_ - HALF, j_});
    w2 = m_mblock.metric.sqrt_det_h_tilde({i_ + HALF, j_});
    h_13_ij1 = m_mblock.metric.h_13({i_ - HALF, j_});
    h_13_ij2 = m_mblock.metric.h_13({i_ + HALF, j_});
    real_t D1_half {(w1 * h_13_ij1 *m_mblock.em(i - 1, j, em::ex1) 
                   + w2 * h_13_ij2 * m_mblock.em(i, j, em::ex1)) 
                   / (w1 + w2)};

    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({i_ - HALF, j_});
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({i_ + HALF, j_});
    beta_ij1      = m_mblock.metric.beta1u({i_ - HALF, j_});
    beta_ij2      = m_mblock.metric.beta1u({i_ + HALF, j_});
    real_t B2_half {(w1 * sqrt_detH_ij1 * beta_ij1 *  m_mblock.em0(i - 1, j, em::bx2) 
                   + w2 * sqrt_detH_ij2 * beta_ij2 * m_mblock.em0(i, j, em::bx2)) 
                   / (w1 + w2)};    
    
    w1 = m_mblock.metric.sqrt_det_h_tilde({i_ - HALF, j_ + HALF});
    w2 = m_mblock.metric.sqrt_det_h_tilde({i_ + HALF, j_ + HALF});
    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({i_ - HALF, j_ + HALF});
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({i_ + HALF, j_ + HALF});
    beta_ij1      = m_mblock.metric.beta1u({i_ - HALF, j_ + HALF});
    beta_ij2      = m_mblock.metric.beta1u({i_ + HALF, j_ + HALF});
    real_t B3_half {(w1 * sqrt_detH_ij1 * beta_ij1 * m_mblock.em0(i - 1, j, em::bx3) 
                   + w2 * sqrt_detH_ij2 * beta_ij2 * m_mblock.em0(i, j, em::bx3)) 
                   / (w1 + w2)};

    h_13_ij1 = m_mblock.metric.h_13({i_, j_});
    h_13_ij2 = m_mblock.metric.h_13({i_ + ONE, j_});
    w1 = m_mblock.metric.sqrt_det_h_tilde({i_, j_});
    w2 = m_mblock.metric.sqrt_det_h_tilde({i_ + ONE, j_});
    real_t D3_half {(w1 * h_13_ij1 *m_mblock.em(i, j, em::ex3) 
                   + w2 * h_13_ij2 * m_mblock.em(i + 1, j, em::ex3)) 
                   / (w1 + w2)};

    real_t D1_cov {h_11_iPj * m_mblock.em(i, j, em::ex1) + D3_half};
    real_t D2_cov {h_22_ijP * m_mblock.em(i, j, em::ex2)};
    real_t D3_cov {h_33_ij * m_mblock.em(i, j, em::ex3) + D1_half};

    m_mblock.aux(i, j, em::ex1) = alpha_iPj * D1_cov;
    m_mblock.aux(i, j, em::ex2) = alpha_ijP * D2_cov - B3_half;
    m_mblock.aux(i, j, em::ex3) = alpha_ij  * D3_cov + B2_half;   
   }

  template <>
  Inline void
  Compute_E<Dimension::THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    // 3d grpic not implemented
  }

   /**
   * Method for computing H. 
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Compute_H0 {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;

  public:
    Compute_H0(const Meshblock<D, SimulationType::GRPIC>& mblock)
      : m_mblock(mblock) {}
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  // First calculation, with B0 and D
  template <>
  Inline void Compute_H0<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(j - N_GHOSTS)};
    index_t j_min {static_cast<index_t>(m_mblock.j_min())};
    index_t j_max {static_cast<index_t>(m_mblock.j_max())};

    real_t h_11_ijP   {m_mblock.metric.h_11({i_, j_ + HALF})};
    real_t h_13_ijP   {m_mblock.metric.h_13({i_, j_ + HALF})};
    real_t h_22_iPj   {m_mblock.metric.h_22({i_ + HALF, j_})};
    real_t h_33_iPjP  {m_mblock.metric.h_33({i_ + HALF, j_ + HALF})};
    real_t h_13_iPjP  {m_mblock.metric.h_13({i_ + HALF, j_ + HALF})};
    real_t alpha_ijP  {m_mblock.metric.alpha({i_, j_ + HALF})};
    real_t alpha_iPj  {m_mblock.metric.alpha({i_ + HALF, j_})};
    real_t alpha_iPjP {m_mblock.metric.alpha({i_ + HALF, j_ + HALF})};
    real_t beta_iPj   {m_mblock.metric.beta1u({i_ + HALF, j_})};
    real_t beta_iPjP  {m_mblock.metric.beta1u({i_ + HALF, j_ + HALF})};
    
    real_t sqrt_detH_iPj  {m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t sqrt_detH_iPjP {m_mblock.metric.sqrt_det_h({i_ + HALF, j_ + HALF})};

    // // D contra interpolation at half cell
    // real_t D2_half {HALF * (m_mblock.em(i, j, em::ex2) + m_mblock.em(i + 1, j , em::ex2))};
    // real_t D3_half {HALF * (m_mblock.em(i, j, em::ex3) + m_mblock.em(i + 1, j , em::ex3))};

    // // B contra interpolation at half cell
    // real_t B1_half {HALF * (m_mblock.em0(i, j, em::bx1) + m_mblock.em0(i + 1, j, em::bx1))};
    // real_t B3_half {HALF * (m_mblock.em0(i - 1, j, em::bx3) + m_mblock.em0(i, j , em::bx3))};

    // // Contravariant B to covariant B
    // real_t B1_cov {h_11_ijP * m_mblock.em0(i, j, em::bx1) + h_13_ijP * B3_half};
    // real_t B2_cov {h_22_iPj * m_mblock.em0(i, j, em::bx2)};
    // real_t B3_cov {h_33_iPjP * m_mblock.em0(i, j, em::bx3) + h_13_iPjP * B1_half};

    // // Compute H_i
    // m_mblock.aux(i, j, em::bx1) = alpha_ijP  * B1_cov;
    // m_mblock.aux(i, j, em::bx2) = alpha_iPj  * B2_cov + sqrt_detH_iPj  * beta_iPj  * D3_half;
    // m_mblock.aux(i, j, em::bx3) = alpha_iPjP * B3_cov - sqrt_detH_iPjP * beta_iPjP * D2_half;

    real_t w1, w2;
    real_t h_13_ij1, h_13_ij2; 
    real_t sqrt_detH_ij1, sqrt_detH_ij2;
    real_t beta_ij1, beta_ij2;

    w1 = m_mblock.metric.sqrt_det_h_tilde({i_, j_ + HALF});
    w2 = m_mblock.metric.sqrt_det_h_tilde({i_ + ONE, j_ + HALF});
    h_13_ij1 = m_mblock.metric.h_13({i_, j_ + HALF});
    h_13_ij2 = m_mblock.metric.h_13({i_ + ONE, j_ + HALF});
    real_t B1_half {(w1 * h_13_ij1 * m_mblock.em0(i, j, em::bx1) 
                   + w2 * h_13_ij2 * m_mblock.em0(i + 1, j, em::bx1)) 
                   / (w1 + w2)};

    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({i_, j_ + HALF});
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({i_ + ONE, j_ + HALF});
    beta_ij1      = m_mblock.metric.beta1u({i_, j_ + HALF});
    beta_ij2      = m_mblock.metric.beta1u({i_ + ONE, j_ + HALF});
    real_t D2_half {(w1 * sqrt_detH_ij1 * beta_ij1 * m_mblock.em(i, j, em::ex2) 
                   + w2 * sqrt_detH_ij2 * beta_ij2 * m_mblock.em(i + 1, j, em::ex2)) 
                   / (w1 + w2)};
    
    w1 = m_mblock.metric.sqrt_det_h_tilde({i_, j_});
    w2 = m_mblock.metric.sqrt_det_h_tilde({i_ + ONE, j_});
    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({i_, j_});
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({i_ + ONE, j_});
    beta_ij1      = m_mblock.metric.beta1u({i_, j_});
    beta_ij2      = m_mblock.metric.beta1u({i_ + ONE, j_});
    real_t D3_half {(w1 * sqrt_detH_ij1 * beta_ij1 * m_mblock.em(i, j, em::ex3) 
                   + w2 * sqrt_detH_ij2 * beta_ij2 * m_mblock.em(i + 1, j, em::ex3)) 
                   / (w1 + w2)};
    
    w1 = m_mblock.metric.sqrt_det_h_tilde({i_ - HALF, j_ + HALF});
    w2 = m_mblock.metric.sqrt_det_h_tilde({i_ + HALF, j_ + HALF});
    h_13_ij1 = m_mblock.metric.h_13({i_ - HALF, j_ + HALF});
    h_13_ij2 = m_mblock.metric.h_13({i_ + HALF, j_ + HALF});
    real_t B3_half {(w1 * h_13_ij1 * m_mblock.em0(i - 1, j, em::bx3) 
                   + w2 * h_13_ij2 * m_mblock.em0(i, j, em::bx3)) 
                   / (w1 + w2)};

    real_t B1_cov {h_11_ijP * m_mblock.em0(i, j, em::bx1) + B3_half};
    real_t B2_cov {h_22_iPj * m_mblock.em0(i, j, em::bx2)};
    real_t B3_cov {h_33_iPjP * m_mblock.em0(i, j, em::bx3) + B1_half};

    m_mblock.aux(i, j, em::bx1) = alpha_ijP  * B1_cov;
    m_mblock.aux(i, j, em::bx2) = alpha_iPj  * B2_cov + D3_half;
    m_mblock.aux(i, j, em::bx3) = alpha_iPjP * B3_cov - D2_half;
   }

  template <>
  Inline void
  Compute_H0<Dimension::THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    // 3d grpic not implemented
  }

  template <Dimension D>
  class Compute_H {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;

  public:
    Compute_H(const Meshblock<D, SimulationType::GRPIC>& mblock)
      : m_mblock(mblock) {}
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  // Second calculation, with B0 and D0
  template <>
  Inline void Compute_H<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(j - N_GHOSTS)};
    index_t j_min {static_cast<index_t>(m_mblock.j_min())};
    index_t j_max {static_cast<index_t>(m_mblock.j_max())};

    real_t h_11_ijP   {m_mblock.metric.h_11({i_, j_ + HALF})};
    real_t h_13_ijP   {m_mblock.metric.h_13({i_, j_ + HALF})};
    real_t h_22_iPj   {m_mblock.metric.h_22({i_ + HALF, j_})};
    real_t h_33_iPjP  {m_mblock.metric.h_33({i_ + HALF, j_ + HALF})};
    real_t h_13_iPjP  {m_mblock.metric.h_13({i_ + HALF, j_ + HALF})};
    real_t alpha_ijP  {m_mblock.metric.alpha({i_, j_ + HALF})};
    real_t alpha_iPj  {m_mblock.metric.alpha({i_ + HALF, j_})};
    real_t alpha_iPjP {m_mblock.metric.alpha({i_ + HALF, j_ + HALF})};
    real_t beta_iPj   {m_mblock.metric.beta1u({i_ + HALF, j_})};
    real_t beta_iPjP  {m_mblock.metric.beta1u({i_ + HALF, j_ + HALF})};

    real_t sqrt_detH_iPj {m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t sqrt_detH_iPjP {m_mblock.metric.sqrt_det_h({i_ + HALF, j_ + HALF})};

    // D contra interpolation at half cell
    // real_t D2_half {HALF * (m_mblock.em0(i, j, em::ex2) + m_mblock.em0(i + 1, j , em::ex2))};
    // real_t D3_half {HALF * (m_mblock.em0(i, j, em::ex3) + m_mblock.em0(i + 1, j , em::ex3))};

    // // B contra interpolation at half cell
    // real_t B1_half {HALF * (m_mblock.em0(i, j, em::bx1) + m_mblock.em0(i + 1, j, em::bx1))};
    // real_t B3_half {HALF * (m_mblock.em0(i - 1, j, em::bx3) + m_mblock.em0(i, j , em::bx3))};

    // // Contravariant B to covariant B
    // real_t B1_cov {h_11_ijP  * m_mblock.em0(i, j, em::bx1) + h_13_ijP * B3_half};
    // real_t B2_cov {h_22_iPj  * m_mblock.em0(i, j, em::bx2)};
    // real_t B3_cov {h_33_iPjP * m_mblock.em0(i, j, em::bx3) + h_13_iPjP * B1_half};

    // // Compute H_i
    // m_mblock.aux(i, j, em::bx1) = alpha_ijP  * B1_cov;
    // m_mblock.aux(i, j, em::bx2) = alpha_iPj  * B2_cov + sqrt_detH_iPj  * beta_iPj  * D3_half;
    // m_mblock.aux(i, j, em::bx3) = alpha_iPjP * B3_cov - sqrt_detH_iPjP * beta_iPjP * D2_half;

    real_t w1, w2;
    real_t h_13_ij1, h_13_ij2; 
    real_t sqrt_detH_ij1, sqrt_detH_ij2;
    real_t beta_ij1, beta_ij2;

    w1 = m_mblock.metric.sqrt_det_h_tilde({i_, j_ + HALF});
    w2 = m_mblock.metric.sqrt_det_h_tilde({i_ + ONE, j_ + HALF});
    h_13_ij1 = m_mblock.metric.h_13({i_, j_ + HALF});
    h_13_ij2 = m_mblock.metric.h_13({i_ + ONE, j_ + HALF});
    real_t B1_half {(w1 * h_13_ij1 * m_mblock.em0(i, j, em::bx1) 
                   + w2 * h_13_ij2 * m_mblock.em0(i + 1, j, em::bx1)) 
                   / (w1 + w2)};

    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({i_, j_ + HALF});
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({i_ + ONE, j_ + HALF});
    beta_ij1      = m_mblock.metric.beta1u({i_, j_ + HALF});
    beta_ij2      = m_mblock.metric.beta1u({i_ + ONE, j_ + HALF});
    real_t D2_half {(w1 * sqrt_detH_ij1 * beta_ij1 * m_mblock.em0(i, j, em::ex2) 
                   + w2 * sqrt_detH_ij2 * beta_ij2 * m_mblock.em0(i + 1, j, em::ex2)) 
                   / (w1 + w2)};
    
    w1 = m_mblock.metric.sqrt_det_h_tilde({i_, j_});
    w2 = m_mblock.metric.sqrt_det_h_tilde({i_ + ONE, j_});
    sqrt_detH_ij1 = m_mblock.metric.sqrt_det_h({i_, j_});
    sqrt_detH_ij2 = m_mblock.metric.sqrt_det_h({i_ + ONE, j_});
    beta_ij1      = m_mblock.metric.beta1u({i_, j_});
    beta_ij2      = m_mblock.metric.beta1u({i_ + ONE, j_});
    real_t D3_half {(w1 * sqrt_detH_ij1 * beta_ij1 * m_mblock.em0(i, j, em::ex3) 
                   + w2 * sqrt_detH_ij2 * beta_ij2 * m_mblock.em0(i + 1, j, em::ex3)) 
                   / (w1 + w2)};
    
    w1 = m_mblock.metric.sqrt_det_h_tilde({i_ - HALF, j_ + HALF});
    w2 = m_mblock.metric.sqrt_det_h_tilde({i_ + HALF, j_ + HALF});
    h_13_ij1 = m_mblock.metric.h_13({i_ - HALF, j_ + HALF});
    h_13_ij2 = m_mblock.metric.h_13({i_ + HALF, j_ + HALF});
    real_t B3_half {(w1 * h_13_ij1 * m_mblock.em0(i - 1, j, em::bx3) 
                   + w2 * h_13_ij2 * m_mblock.em0(i, j, em::bx3)) 
                   / (w1 + w2)};

    real_t B1_cov {h_11_ijP  * m_mblock.em0(i, j, em::bx1) + B3_half};
    real_t B2_cov {h_22_iPj  * m_mblock.em0(i, j, em::bx2)};
    real_t B3_cov {h_33_iPjP * m_mblock.em0(i, j, em::bx3) + B1_half};

    m_mblock.aux(i, j, em::bx1) = alpha_ijP  * B1_cov;
    m_mblock.aux(i, j, em::bx2) = alpha_iPj  * B2_cov + D3_half;
    m_mblock.aux(i, j, em::bx3) = alpha_iPjP * B3_cov - D2_half;
   }

  template <>
  Inline void
  Compute_H<Dimension::THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    // 3d grpic not implemented
  }
  /**
   * Time average B and D field
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Average_EM {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;

  public:
    Average_EM(const Meshblock<D, SimulationType::GRPIC>& mblock)
      : m_mblock(mblock) {}
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  template <>
  Inline void Average_EM<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {

    m_mblock.em0(i, j, em::bx1) = HALF * (m_mblock.em0(i, j, em::bx1) + m_mblock.em(i, j, em::bx1));
    m_mblock.em0(i, j, em::bx2) = HALF * (m_mblock.em0(i, j, em::bx2) + m_mblock.em(i, j, em::bx2));
    m_mblock.em0(i, j, em::bx3) = HALF * (m_mblock.em0(i, j, em::bx3) + m_mblock.em(i, j, em::bx3));
    m_mblock.em0(i, j, em::ex1) = HALF * (m_mblock.em0(i, j, em::ex1) + m_mblock.em(i, j, em::ex1));
    m_mblock.em0(i, j, em::ex2) = HALF * (m_mblock.em0(i, j, em::ex2) + m_mblock.em(i, j, em::ex2));
    m_mblock.em0(i, j, em::ex3) = HALF * (m_mblock.em0(i, j, em::ex3) + m_mblock.em(i, j, em::ex3));
   }
  
  template <>
  Inline void
  Average_EM<Dimension::THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    // 3d grpic not implemented
  }
  /**
   * Time average currents
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Average_J {
    using index_t = typename RealFieldND<D, 3>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;

  public:
    Average_J(const Meshblock<D, SimulationType::GRPIC>& mblock)
      : m_mblock(mblock) {}
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  template <>
  Inline void Average_J<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {

    m_mblock.cur(i, j, cur::jx1) = HALF * (m_mblock.cur0(i, j, cur::jx1) + m_mblock.cur(i, j, cur::jx1));
    m_mblock.cur(i, j, cur::jx2) = HALF * (m_mblock.cur0(i, j, cur::jx2) + m_mblock.cur(i, j, cur::jx2));
    m_mblock.cur(i, j, cur::jx3) = HALF * (m_mblock.cur0(i, j, cur::jx3) + m_mblock.cur(i, j, cur::jx3));
   }
  
  template <>
  Inline void
  Average_J<Dimension::THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    // 3d grpic not implemented
  }

} // namespace ntt

#endif
