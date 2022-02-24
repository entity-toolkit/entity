#ifndef GRPIC_FARADAY_H
#define GRPIC_FARADAY_H

#include "global.h"
#include "fields.h"
#include "meshblock.h"
#include "grpic.h"

#include <stdexcept>

namespace ntt {

  /**
   * Algorithms for Faraday's law: `dB/dt = -curl E`.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Faraday_push0{
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t m_coeff;

  public:
    Faraday_push0(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff) {}
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  // First push, updates B0.
  template <>
  Inline void Faraday_push0<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(j - N_GHOSTS)};
    index_t j_min {static_cast<index_t>(m_mblock.j_min())};
    index_t j_max {static_cast<index_t>(m_mblock.j_max() - 2)};

    real_t inv_sqrt_detH_iPj  {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_ijP  {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    real_t inv_sqrt_detH_iPjP {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_ + HALF})};
   
    m_mblock.em0(i, j, em::bx1)
      += m_coeff * inv_sqrt_detH_ijP * (m_mblock.aux(i, j, em::ex3) - m_mblock.aux(i, j + 1, em::ex3));

    if ((j == j_min) || (j == j_max)) {
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
  Inline void
  Faraday_push0<Dimension::THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    // 3d curvilinear faraday not implemented
  }

  template <Dimension D>
  class Faraday_push {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t m_coeff;

  public:
    Faraday_push(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff) {}
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  // Second push, updates B but assigns it to B0.
  template <>
  Inline void Faraday_push<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(j - N_GHOSTS)};
    index_t j_min {static_cast<index_t>(m_mblock.j_min())};
    index_t j_max {static_cast<index_t>(m_mblock.j_max() - 2)};

    real_t inv_sqrt_detH_iPj  {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_ijP  {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    real_t inv_sqrt_detH_iPjP {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_ + HALF})};
   
    m_mblock.em0(i, j, em::bx1) = m_mblock.em(i, j, em::bx1) + 
                                  m_coeff * inv_sqrt_detH_ijP * (m_mblock.aux(i, j, em::ex3) - m_mblock.aux(i, j + 1, em::ex3));

    if ((j == j_min) || (j == j_max)) {
    m_mblock.em0(i, j, em::bx2) = ZERO;
    } else {
    m_mblock.em0(i, j, em::bx2) = m_mblock.em(i, j, em::bx2) + 
                                  m_coeff * inv_sqrt_detH_iPj * (m_mblock.aux(i + 1, j, em::ex3) - m_mblock.aux(i, j, em::ex3));
    }

    m_mblock.em0(i, j, em::bx3) = m_mblock.em(i, j, em::bx3) 
                                  +  m_coeff * inv_sqrt_detH_iPjP
                                  * (m_mblock.aux(i, j + 1, em::ex1) - m_mblock.aux(i, j, em::ex1)
                                  +  m_mblock.aux(i, j, em::ex2) - m_mblock.aux(i + 1, j, em::ex2));
    }

  template <>
  Inline void
  Faraday_push<Dimension::THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    // 3d curvilinear faraday not implemented
  }

} // namespace ntt

#endif
