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

    real_t hrr_iPj {m_mblock.metric.h_11({i_ + HALF, j_})};
    real_t hrph_iPj {m_mblock.metric.h_13({i_ + HALF, j_})};
    real_t hthth_ijP {m_mblock.metric.h_22({i_, j_ + HALF})};
    real_t hphph_ij {m_mblock.metric.h_33({i_, j_})};
    real_t hrph_ij {m_mblock.metric.h_13({i_, j_})};
    real_t alpha_ij {m_mblock.metric.alpha({i_, j_})};
    real_t alpha_iPj {m_mblock.metric.alpha({i_ + HALF, j_})};
    real_t alpha_ijP {m_mblock.metric.alpha({i_, j_ + HALF})};
    real_t beta_ij {m_mblock.metric.betar({i_, j_})};
    real_t beta_ijP {m_mblock.metric.betar({i_, j_ + HALF})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    real_t inv_sqrt_detH_ij {ONE / m_mblock.metric.sqrt_det_h({i_, j_})};

    // B contra interpolation at half cell
    real_t Bth_half {HALF * (m_mblock.em(i - 1, j, em::bx2) + m_mblock.em(i, j, em::bx2))};
    real_t Bph_half {HALF * (m_mblock.em(i - 1, j, em::bx3) + m_mblock.em(i, j, em::bx3))};

    // D contra interpolation at half cell
    real_t Dr_half {HALF * (m_mblock.em0(i - 1, j, em::ex1) + m_mblock.em0(i, j, em::ex1))};
    real_t Dph_half {HALF * (m_mblock.em0(i, j, em::ex3) + m_mblock.em0(i + 1, j, em::ex3))};

    // Contravariant D to covariant D
    real_t Dr_cov {hrr_iPj * m_mblock.em0(i, j, em::ex1) + hrph_iPj * Dph_half};
    real_t Dth_cov {hthth_ijP * m_mblock.em0(i, j, em::ex2)};
    real_t Dph_cov {hphph_ij * m_mblock.em0(i, j, em::ex3) + hrph_ij * Dr_half};

    // Compute E_i
    m_mblock.aux(i, j, em::ex1) = alpha_iPj * Dr_cov;
    m_mblock.aux(i, j, em::ex2) = alpha_ijP * Dth_cov - inv_sqrt_detH_ijP * beta_ijP *  Bph_half;
    
    if ((j == j_min) || (j == j_max)) {
    m_mblock.aux(i, j, em::ex3) = ZERO;
    } else {
    m_mblock.aux(i, j, em::ex3) = alpha_ij * Dph_cov + inv_sqrt_detH_ij * beta_ij *  Bth_half;;
    }

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
    
    real_t hrr_iPj {m_mblock.metric.h_11({i_ + HALF, j_})};
    real_t hrph_iPj {m_mblock.metric.h_13({i_ + HALF, j_})};
    real_t hthth_ijP {m_mblock.metric.h_22({i_, j_ + HALF})};
    real_t hphph_ij {m_mblock.metric.h_33({i_, j_})};
    real_t hrph_ij {m_mblock.metric.h_13({i_, j_})};
    real_t alpha_ij {m_mblock.metric.alpha({i_, j_})};
    real_t alpha_iPj {m_mblock.metric.alpha({i_ + HALF, j_})};
    real_t alpha_ijP {m_mblock.metric.alpha({i_, j_ + HALF})};
    real_t beta_ij {m_mblock.metric.betar({i_, j_})};
    real_t beta_ijP {m_mblock.metric.betar({i_, j_ + HALF})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    real_t inv_sqrt_detH_ij {ONE / m_mblock.metric.sqrt_det_h({i_, j_})};

    // B contra interpolation at half cell
    real_t Bth_half {HALF * (m_mblock.em0(i - 1, j, em::bx2) + m_mblock.em0(i, j, em::bx2))};
    real_t Bph_half {HALF * (m_mblock.em0(i - 1, j, em::bx3) + m_mblock.em0(i, j, em::bx3))};

    // D contra interpolation at half cell
    real_t Dr_half {HALF * (m_mblock.em(i - 1, j, em::ex1) + m_mblock.em(i, j, em::ex1))};
    real_t Dph_half {HALF * (m_mblock.em(i, j, em::ex3) + m_mblock.em(i + 1, j, em::ex3))};

    // Contravariant D to covariant D
    real_t Dr_cov {hrr_iPj * m_mblock.em(i, j, em::ex1) + hrph_iPj * Dph_half};
    real_t Dth_cov {hthth_ijP * m_mblock.em(i, j, em::ex2)};
    real_t Dph_cov {hphph_ij * m_mblock.em(i, j, em::ex3) + hrph_ij * Dr_half};

    // Compute E_i
    m_mblock.aux(i, j, em::ex1) = alpha_iPj * Dr_cov;
    m_mblock.aux(i, j, em::ex2) = alpha_ijP * Dth_cov - inv_sqrt_detH_ijP * beta_ijP *  Bph_half;
    
    if ((j == j_min) || (j == j_max)) {
    m_mblock.aux(i, j, em::ex3) = ZERO;
    } else {
    m_mblock.aux(i, j, em::ex3) = alpha_ij * Dph_cov + inv_sqrt_detH_ij * beta_ij *  Bth_half;;
    }

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

    real_t hrr_ijP {m_mblock.metric.h_11({i_, j_ + HALF})};
    real_t hrph_ijP {m_mblock.metric.h_13({i_, j_ + HALF})};
    real_t hthth_iPj {m_mblock.metric.h_22({i_ + HALF, j_})};
    real_t hphph_iPjP {m_mblock.metric.h_33({i_ + HALF, j_ + HALF})};
    real_t hrph_iPjP {m_mblock.metric.h_13({i_ + HALF, j_ + HALF})};
    real_t alpha_ijP {m_mblock.metric.alpha({i_, j_ + HALF})};
    real_t alpha_iPj {m_mblock.metric.alpha({i_ + HALF, j_})};
    real_t alpha_iPjP {m_mblock.metric.alpha({i_ + HALF, j_ + HALF})};
    real_t beta_iPj {m_mblock.metric.betar({i_ + HALF, j_})};
    real_t beta_iPjP {m_mblock.metric.betar({i_ + HALF, j_ + HALF})};
    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_iPjP {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_ + HALF})};

    // D contra interpolation at half cell
    real_t Dth_half {HALF * (m_mblock.em(i, j, em::ex2) + m_mblock.em(i + 1, j , em::ex2))};
    real_t Dph_half {HALF * (m_mblock.em(i, j , em::ex3) + m_mblock.em(i + 1, j , em::ex3))};

    // B contra interpolation at half cell
    real_t Br_half {HALF * (m_mblock.em0(i, j, em::bx1) + m_mblock.em0(i + 1, j, em::bx1))};
    real_t Bph_half {HALF * (m_mblock.em0(i - 1, j, em::bx3) + m_mblock.em0(i, j , em::bx3))};

    // Contravariant B to covariant B
    real_t Br_cov {hrr_ijP * m_mblock.em0(i, j, em::bx1) + hrph_ijP * Bph_half};
    real_t Bth_cov {hthth_iPj * m_mblock.em0(i, j, em::bx2)};
    real_t Bph_cov {hphph_iPjP * m_mblock.em0(i, j, em::bx3) + hrph_iPjP * Br_half};

    // Compute H_i
    m_mblock.aux(i, j, em::bx1) = alpha_ijP * Br_cov;

    if ((j == j_min) || (j == j_max)) {
    m_mblock.aux(i, j, em::bx2) = ZERO;
    } else {
    m_mblock.aux(i, j, em::bx2) = alpha_iPj * Bth_cov + inv_sqrt_detH_iPj * beta_iPj *  Dph_half;
    }

    // std::printf("%f %f %f  %lu %lu \n", m_mblock.aux(i, j, em::bx1), m_mblock.aux(i, j, em::bx2), alpha_iPj * Bth_cov + inv_sqrt_detH_iPj * beta_iPj *  Dph_half, i, j);

    m_mblock.aux(i, j, em::bx3) = alpha_iPjP * Bph_cov - inv_sqrt_detH_iPjP * beta_iPjP *  Dth_half;
  
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

    real_t hrr_ijP {m_mblock.metric.h_11({i_, j_ + HALF})};
    real_t hrph_ijP {m_mblock.metric.h_13({i_, j_ + HALF})};
    real_t hthth_iPj {m_mblock.metric.h_22({i_ + HALF, j_})};
    real_t hphph_iPjP {m_mblock.metric.h_33({i_ + HALF, j_ + HALF})};
    real_t hrph_iPjP {m_mblock.metric.h_13({i_ + HALF, j_ + HALF})};
    real_t alpha_ijP {m_mblock.metric.alpha({i_, j_ + HALF})};
    real_t alpha_iPj {m_mblock.metric.alpha({i_ + HALF, j_})};
    real_t alpha_iPjP {m_mblock.metric.alpha({i_ + HALF, j_ + HALF})};
    real_t beta_iPj {m_mblock.metric.betar({i_ + HALF, j_})};
    real_t beta_iPjP {m_mblock.metric.betar({i_ + HALF, j_ + HALF})};
    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_iPjP {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_ + HALF})};

    // D contra interpolation at half cell
    real_t Dth_half {HALF * (m_mblock.em0(i, j, em::ex2) + m_mblock.em0(i + 1, j , em::ex2))};
    real_t Dph_half {HALF * (m_mblock.em0(i, j , em::ex3) + m_mblock.em0(i + 1, j , em::ex3))};

    // B contra interpolation at half cell
    real_t Br_half {HALF * (m_mblock.em0(i, j, em::bx1) + m_mblock.em0(i + 1, j, em::bx1))};
    real_t Bph_half {HALF * (m_mblock.em0(i - 1, j, em::bx3) + m_mblock.em0(i, j , em::bx3))};

    // Contravariant B to covariant B
    real_t Br_cov {hrr_ijP * m_mblock.em0(i, j, em::bx1) + hrph_ijP * Bph_half};
    real_t Bth_cov {hthth_iPj * m_mblock.em0(i, j, em::bx2)};
    real_t Bph_cov {hphph_iPjP * m_mblock.em0(i, j, em::bx3) + hrph_iPjP * Br_half};

    // Compute H_i
    m_mblock.aux(i, j, em::bx1) = alpha_ijP * Br_cov;

    if ((j == j_min) || (j == j_max)) {
    m_mblock.aux(i, j, em::bx2) = ZERO;
    } else {
    m_mblock.aux(i, j, em::bx2) = alpha_iPj * Bth_cov + inv_sqrt_detH_iPj * beta_iPj *  Dph_half;
    }
  
    m_mblock.aux(i, j, em::bx3) = alpha_iPjP * Bph_cov - inv_sqrt_detH_iPjP * beta_iPjP *  Dth_half;
   }

  template <>
  Inline void
  Compute_H<Dimension::THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    // 3d grpic not implemented
  }
  /**
   * Time average B and D field: is it the most efficient way?
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
