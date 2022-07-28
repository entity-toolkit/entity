#ifndef PIC_FARADAY_CURVILINEAR_H
#define PIC_FARADAY_CURVILINEAR_H

#include "global.h"
#include "fields.h"
#include "meshblock.h"
#include "pic.h"

#include <stdexcept>

namespace ntt {

  /**
   * @brief Algorithm for the Faraday's law: `dB/dt = -curl E` in Curvilinear space (diagonal metric).
   * @tparam D Dimension.
   */
  template <Dimension D>
  class FaradayCurvilinear {
    Meshblock<D, SimulationType::PIC> m_mblock;
    real_t                            m_coeff;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param coeff Coefficient to be multiplied by dB/dt = coeff * -curl E.
     */
    FaradayCurvilinear(const Meshblock<D, SimulationType::PIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff) {}
    
    /**
     * @brief 2D implementation of the algorithm.
     * @param i1 index.
     * @param i2 index.
     */
    Inline void operator()(index_t, index_t) const;
    /**
     * @brief 3D implementation of the algorithm.
     * @param i1 index.
     * @param i2 index.
     * @param i3 index.
     */
    Inline void operator()(index_t, index_t, index_t) const;
  };

  template <>
  Inline void FaradayCurvilinear<Dimension::TWO_D>::operator()(index_t i,
                                                               index_t j) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};

    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    real_t inv_sqrt_detH_iPjP {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_ + HALF})};
    real_t h1_iPjP1 {m_mblock.metric.h_11({i_ + HALF, j_ + ONE})};
    real_t h1_iPj {m_mblock.metric.h_11({i_ + HALF, j_})};
    real_t h2_iP1jP {m_mblock.metric.h_22({i_ + ONE, j_ + HALF})};
    real_t h2_ijP {m_mblock.metric.h_22({i_, j_ + HALF})};
    real_t h3_ij {m_mblock.metric.h_33({i_, j_})};
    real_t h3_iP1j {m_mblock.metric.h_33({i_ + ONE, j_})};
    real_t h3_ijP1 {m_mblock.metric.h_33({i_, j_ + ONE})};

    m_mblock.em(i, j, em::bx1)
      += m_coeff * inv_sqrt_detH_ijP
         * (h3_ij * m_mblock.em(i, j, em::ex3) - h3_ijP1 * m_mblock.em(i, j + 1, em::ex3));
    m_mblock.em(i, j, em::bx2)
      += m_coeff * inv_sqrt_detH_iPj
         * (h3_iP1j * m_mblock.em(i + 1, j, em::ex3) - h3_ij * m_mblock.em(i, j, em::ex3));
    m_mblock.em(i, j, em::bx3)
      += m_coeff * inv_sqrt_detH_iPjP
         * (h1_iPjP1 * m_mblock.em(i, j + 1, em::ex1) - h1_iPj * m_mblock.em(i, j, em::ex1)
            + h2_ijP * m_mblock.em(i, j, em::ex2) - h2_iP1jP * m_mblock.em(i + 1, j, em::ex2));
  }

  template <>
  Inline void FaradayCurvilinear<Dimension::THREE_D>::operator()(index_t,
                                                                 index_t,
                                                                 index_t) const {
    // 3d curvilinear faraday not implemented
  }
} // namespace ntt

#endif
