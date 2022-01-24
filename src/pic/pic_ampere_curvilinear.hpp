#ifndef PIC_AMPERE_CURVILINEAR_H
#define PIC_AMPERE_CURVILINEAR_H

#include "global.h"
#include "fields.h"
#include "meshblock.h"
#include "pic.h"

namespace ntt {

  /**
   * Algorithm for the Ampere's law: `dE/dt = curl B` in curvilinear space.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class AmpereCurvilinear {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::PIC> m_mblock;
    real_t m_coeff;

  public:
    AmpereCurvilinear(const Meshblock<D, SimulationType::PIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff) {}
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  template <>
  Inline void AmpereCurvilinear<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(j - N_GHOSTS)};

    real_t inv_sqrt_detH_ij {ONE / m_mblock.metric.sqrt_det_h({i_, j_})};
    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    real_t h1_ijM {m_mblock.metric.h_11({i_, j_ - HALF})};
    real_t h1_ijP {m_mblock.metric.h_11({i_, j_ + HALF})};
    real_t h2_iPj {m_mblock.metric.h_22({i_ + HALF, j_})};
    real_t h2_iMj {m_mblock.metric.h_22({i_ - HALF, j_})};
    real_t h3_iMjP {m_mblock.metric.h_33({i_ - HALF, j_ + HALF})};
    real_t h3_iPjM {m_mblock.metric.h_33({i_ + HALF, j_ - HALF})};
    real_t h3_iPjP {m_mblock.metric.h_33({i_ + HALF, j_ + HALF})};

    m_mblock.em(i, j, em::ex1) += m_coeff * inv_sqrt_detH_iPj
                                  * (h3_iPjP * m_mblock.em(i, j, em::bx3) - h3_iPjM * m_mblock.em(i, j - 1, em::bx3));
    m_mblock.em(i, j, em::ex2) += m_coeff * inv_sqrt_detH_ijP
                                  * (h3_iMjP * m_mblock.em(i - 1, j, em::bx3) - h3_iPjP * m_mblock.em(i, j, em::bx3));
    m_mblock.em(i, j, em::ex3) += m_coeff * inv_sqrt_detH_ij
                                  * (h1_ijM * m_mblock.em(i, j - 1, em::bx1) - h1_ijP * m_mblock.em(i, j, em::bx1)
                                     + h2_iPj * m_mblock.em(i, j, em::bx2) - h2_iMj * m_mblock.em(i - 1, j, em::bx2));
  }

  template <>
  Inline void
  AmpereCurvilinear<Dimension::THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    // 3d curvilinear ampere not implemented
  }

  /**
   * Algorithm for the Ampere's law: `dE/dt = curl B` in curvilinear space near the polar axes (integral form).
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class AmpereCurvilinearPoles {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::PIC> m_mblock;
    real_t m_coeff;
    int m_nj;

  public:
    AmpereCurvilinearPoles(const Meshblock<D, SimulationType::PIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff), m_nj(m_mblock.Nj()) {}
    Inline void operator()(const index_t) const;
  };

  template <>
  Inline void AmpereCurvilinearPoles<Dimension::TWO_D>::operator()(const index_t i) const {
    index_t j_min {N_GHOSTS};
    index_t j_max {static_cast<index_t>(m_nj) + N_GHOSTS - 1};

    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t j_max_ {static_cast<real_t>(j_max - N_GHOSTS)};

    real_t inv_polar_area_iPj {ONE / m_mblock.metric.polar_area({i_ + HALF, HALF})};
    real_t h3_min_iPjP {m_mblock.metric.h_33({i_ + HALF, HALF})};
    real_t h3_max_iPjP {m_mblock.metric.h_33({i_ + HALF, j_max_ + HALF})};

    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, HALF})};
    real_t h3_min_iMjP {m_mblock.metric.h_33({i_ - HALF, HALF})};

    // theta = 0
    m_mblock.em(i, j_min, em::ex1) += inv_polar_area_iPj * m_coeff * (h3_min_iPjP * m_mblock.em(i, j_min, em::bx3));
    // theta = pi
    m_mblock.em(i, j_max + 1, em::ex1) -= inv_polar_area_iPj * m_coeff * (h3_max_iPjP * m_mblock.em(i, j_max, em::bx3));

    // j = jmin + 1/2
    m_mblock.em(i, j_min, em::ex2)
      += inv_sqrt_detH_ijP * m_coeff
         * (h3_min_iMjP * m_mblock.em(i - 1, j_min, em::bx3) - h3_min_iPjP * m_mblock.em(i, j_min, em::bx3));
  }

} // namespace ntt

#endif
