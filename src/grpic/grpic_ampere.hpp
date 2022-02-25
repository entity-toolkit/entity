#ifndef GRPIC_AMPERE_H
#define GRPIC_AMPERE_H

#include "global.h"
#include "fields.h"
#include "meshblock.h"
#include "grpic.h"

namespace ntt {

  /**
   * Algorithms for Ampere's law: `dD/dt = curl H`.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Ampere_push0 {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t m_coeff;

  public:
    Ampere_push0(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff) {}
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  // First push, updates D0 with J.
  template <>
  Inline void Ampere_push0<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(j - N_GHOSTS)};

    real_t inv_sqrt_detH_ij {ONE / m_mblock.metric.sqrt_det_h({i_, j_})};
    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};

    m_mblock.em0(i, j, em::ex1) += m_coeff * inv_sqrt_detH_iPj 
                                  * (m_mblock.aux(i, j, em::bx3) - m_mblock.aux(i, j - 1, em::bx3));
                                  // - TWO * TWO_PI * m_coeff * m_mblock.cur(i, j, cur::jx1);
    m_mblock.em0(i, j, em::ex2) += m_coeff * inv_sqrt_detH_ijP
                                  * (m_mblock.aux(i - 1, j, em::bx3) - m_mblock.aux(i, j, em::bx3));
                                  // - TWO * TWO_PI * m_coeff * m_mblock.cur(i, j, cur::jx2);
    m_mblock.em0(i, j, em::ex3) += m_coeff * inv_sqrt_detH_ij
                                  * (m_mblock.aux(i, j - 1, em::bx1) - m_mblock.aux(i, j, em::bx1)
                                  + m_mblock.aux(i, j, em::bx2) - m_mblock.aux(i - 1, j, em::bx2));
                                  // - TWO * TWO_PI * m_coeff * m_mblock.cur(i, j, cur::jx3);
  }
  
  template <>
  Inline void
  Ampere_push0<Dimension::THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    // 3d curvilinear ampere not implemented
  }

  template <Dimension D>
  class Ampere_push {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t m_coeff;

  public:
    Ampere_push(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff) {}
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  // Second push, updates D with J0 but assigns it to D0.
  template <>
  Inline void Ampere_push<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(j - N_GHOSTS)};

    real_t inv_sqrt_detH_ij {ONE / m_mblock.metric.sqrt_det_h({i_, j_})};
    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};

    m_mblock.em0(i, j, em::ex1) = m_mblock.em(i, j, em::ex1) + m_coeff * inv_sqrt_detH_iPj 
                                  * (m_mblock.aux(i, j, em::bx3) - m_mblock.aux(i, j - 1, em::bx3));
                                  // - TWO * TWO_PI * m_coeff * m_mblock.cur0(i, j, cur::jx1);
    m_mblock.em0(i, j, em::ex2) = m_mblock.em(i, j, em::ex2) + m_coeff * inv_sqrt_detH_ijP
                                  * (m_mblock.aux(i - 1, j, em::bx3) - m_mblock.aux(i, j, em::bx3));
                                  // - TWO * TWO_PI * m_coeff * m_mblock.cur0(i, j, cur::jx2);
    m_mblock.em0(i, j, em::ex3) = m_mblock.em(i, j, em::ex3) + m_coeff * inv_sqrt_detH_ij
                                  * (m_mblock.aux(i, j - 1, em::bx1) - m_mblock.aux(i, j, em::bx1)
                                  + m_mblock.aux(i, j, em::bx2) - m_mblock.aux(i - 1, j, em::bx2));
                                  // - TWO * TWO_PI * m_coeff * m_mblock.cur0(i, j, cur::jx3);
  }

  template <>
  Inline void
  Ampere_push<Dimension::THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    // 3d curvilinear ampere not implemented
  }

  template <Dimension D>
  class Ampere_push_initial {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t m_coeff;

  public:
    Ampere_push_initial(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff) {}
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  // Second push, updates D with J0 but assigns it to D0.
  template <>
  Inline void Ampere_push_initial<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(j - N_GHOSTS)};

    real_t inv_sqrt_detH_ij {ONE / m_mblock.metric.sqrt_det_h({i_, j_})};
    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};

    m_mblock.em(i, j, em::ex1) += m_coeff * inv_sqrt_detH_iPj 
                                  * (m_mblock.aux(i, j, em::bx3) - m_mblock.aux(i, j - 1, em::bx3));
                                  // - TWO * TWO_PI * m_coeff * m_mblock.cur(i, j, cur::jx1);
    m_mblock.em(i, j, em::ex2) += m_coeff * inv_sqrt_detH_ijP
                                  * (m_mblock.aux(i - 1, j, em::bx3) - m_mblock.aux(i, j, em::bx3));
                                  // - TWO * TWO_PI * m_coeff * m_mblock.cur(i, j, cur::jx2);
    m_mblock.em(i, j, em::ex3) += m_coeff * inv_sqrt_detH_ij
                                  * (m_mblock.aux(i, j - 1, em::bx1) - m_mblock.aux(i, j, em::bx1)
                                  + m_mblock.aux(i, j, em::bx2) - m_mblock.aux(i - 1, j, em::bx2));
                                  // - TWO * TWO_PI * m_coeff * m_mblock.cur(i, j, cur::jx3);
  }

  template <>
  Inline void
  Ampere_push_initial<Dimension::THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    // 3d curvilinear ampere not implemented
  }

  /**
   * Algorithms for Ampere's law: `dD/dt = curl H` near the polar axes (integral form).
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Ampere_Poles0 {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t m_coeff;
    std::size_t m_nj;

  public:
    Ampere_Poles0(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff), m_nj(m_mblock.Nj()) {}
    Inline void operator()(const index_t) const;
  };

  // First push, updates D0 with J.
  template <>
  Inline void Ampere_Poles0<Dimension::TWO_D>::operator()(const index_t i) const {
    index_t j_min {N_GHOSTS};
    index_t j_max {static_cast<index_t>(m_nj) + N_GHOSTS - 1};
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};

    real_t inv_polar_area_iPj {ONE / m_mblock.metric.polar_area({i_ + HALF, HALF})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, HALF})};

    // theta = 0
    m_mblock.em0(i, j_min, em::ex1) += inv_polar_area_iPj * m_coeff * (m_mblock.aux(i, j_min, em::bx3));
                                     // - TWO * TWO_PI * m_coeff * m_mblock.cur(i, j_min, cur::jx1);
    // theta = pi
    m_mblock.em0(i, j_max + 1, em::ex1) -= inv_polar_area_iPj * m_coeff * (m_mblock.aux(i, j_max, em::bx3));
                                         // - TWO * TWO_PI * m_coeff * m_mblock.cur(i, j_max + 1, cur::jx2);

    // j = jmin + 1/2
    m_mblock.em0(i, j_min, em::ex2)
      += inv_sqrt_detH_ijP * m_coeff * (m_mblock.aux(i - 1, j_min, em::bx3) - m_mblock.aux(i, j_min, em::bx3));
      // - TWO * TWO_PI * m_coeff * m_mblock.cur(i, j_min, cur::jx2);
  }

  template <Dimension D>
  class Ampere_Poles {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t m_coeff;
    std::size_t m_nj;

  public:
    Ampere_Poles(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff), m_nj(m_mblock.Nj()) {}
    Inline void operator()(const index_t) const;
  };

  // Second push, updates D with J0 but assigns it to D0.
  template <>
  Inline void Ampere_Poles<Dimension::TWO_D>::operator()(const index_t i) const {
    index_t j_min {N_GHOSTS};
    index_t j_max {static_cast<index_t>(m_nj) + N_GHOSTS - 1};
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};

    real_t inv_polar_area_iPj {ONE / m_mblock.metric.polar_area({i_ + HALF, HALF})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, HALF})};

    // theta = 0
    m_mblock.em0(i, j_min, em::ex1) = m_mblock.em(i, j_min, em::ex1) + inv_polar_area_iPj * m_coeff * (m_mblock.aux(i, j_min, em::bx3));
                                      //- TWO * TWO_PI * m_coeff * m_mblock.cur0(i, j_min, cur::jx1);
    // theta = pi
    m_mblock.em0(i, j_max + 1, em::ex1) = m_mblock.em(i, j_max + 1, em::ex1) - inv_polar_area_iPj * m_coeff * (m_mblock.aux(i, j_max, em::bx3));
                                          // - TWO * TWO_PI * m_coeff * m_mblock.cur0(i, j_max + 1, cur::jx2);

    // j = jmin + 1/2
    m_mblock.em0(i, j_min, em::ex2) = m_mblock.em(i, j_min, em::ex2) 
                                    + inv_sqrt_detH_ijP * m_coeff * (m_mblock.aux(i - 1, j_min, em::bx3) - m_mblock.aux(i, j_min, em::bx3));
                                    // - TWO * TWO_PI * m_coeff * m_mblock.cur0(i, j_min, cur::jx2);
    }

  template <Dimension D>
  class Ampere_Poles_initial {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t m_coeff;
    std::size_t m_nj;

  public:
    Ampere_Poles_initial(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff), m_nj(m_mblock.Nj()) {}
    Inline void operator()(const index_t) const;
  };

  // Second push, updates D with J0 but assigns it to D0.
  template <>
  Inline void Ampere_Poles_initial<Dimension::TWO_D>::operator()(const index_t i) const {
    index_t j_min {N_GHOSTS};
    index_t j_max {static_cast<index_t>(m_nj) + N_GHOSTS - 1};
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};

    real_t inv_polar_area_iPj {ONE / m_mblock.metric.polar_area({i_ + HALF, HALF})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, HALF})};

    // theta = 0
    m_mblock.em(i, j_min, em::ex1) += inv_polar_area_iPj * m_coeff * (m_mblock.aux(i, j_min, em::bx3));
                                     // - TWO * TWO_PI * m_coeff * m_mblock.cur(i, j_min, cur::jx1);
    // theta = pi
    m_mblock.em(i, j_max + 1, em::ex1) -= inv_polar_area_iPj * m_coeff * (m_mblock.aux(i, j_max, em::bx3));
                                         // - TWO * TWO_PI * m_coeff * m_mblock.cur(i, j_max + 1, cur::jx2);

    // j = jmin + 1/2
    m_mblock.em(i, j_min, em::ex2)
      += inv_sqrt_detH_ijP * m_coeff * (m_mblock.aux(i - 1, j_min, em::bx3) - m_mblock.aux(i, j_min, em::bx3));
      // - TWO * TWO_PI * m_coeff * m_mblock.cur(i, j_min, cur::jx2);
  }


} // namespace ntt

#endif
