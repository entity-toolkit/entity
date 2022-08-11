#ifndef PIC_ADD_CURRENTS_MINKOWSKI_H
#define PIC_ADD_CURRENTS_MINKOWSKI_H

#include "global.h"
#include "pic.h"

// !TODO: Wrong treatment of the axes

namespace ntt {
  /**
   * @brief Add the currents to the E field.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class AddCurrentsMinkowski {
    Meshblock<D, SimulationType::PIC> m_mblock;
    real_t                            m_coeff;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     */
    AddCurrentsMinkowski(const Meshblock<D, SimulationType::PIC>& mblock, const real_t& coeff)
      : m_mblock {mblock}, m_coeff {coeff} {}
    /**
     * @brief 1D version of the add current.
     * @param i1 index.
     */
    Inline void operator()(index_t i1) const;
    /**
     * @brief 2D version of the add current.
     * @param i1 index.
     * @param i2 index.
     */
    Inline void operator()(index_t i1, index_t i2) const;
    /**
     * @brief 3D version of the add current.
     * @param i1 index.
     * @param i2 index.
     * @param i3 index.
     */
    Inline void operator()(index_t i1, index_t i2, index_t i3) const;
  };

  template <>
  Inline void AddCurrentsMinkowski<Dimension::ONE_D>::operator()(index_t i) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t inv_sqrt_detH_i {ONE / m_mblock.metric.sqrt_det_h({i_})};
    real_t inv_sqrt_detH_iP {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF})};
    m_mblock.em(i, em::ex1) += m_coeff * m_mblock.cur(i, cur::jx1) * inv_sqrt_detH_iP;
    m_mblock.em(i, em::ex2) += m_coeff * m_mblock.cur(i, cur::jx2) * inv_sqrt_detH_i;
    m_mblock.em(i, em::ex3) += m_coeff * m_mblock.cur(i, cur::jx3) * inv_sqrt_detH_i;
  }

  template <>
  Inline void AddCurrentsMinkowski<Dimension::TWO_D>::operator()(index_t i, index_t j) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};
    real_t inv_sqrt_detH_ij {ONE / m_mblock.metric.sqrt_det_h({i_, j_})};
    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    m_mblock.em(i, j, em::ex1) += m_coeff * m_mblock.cur(i, j, cur::jx1) * inv_sqrt_detH_iPj;
    m_mblock.em(i, j, em::ex2) += m_coeff * m_mblock.cur(i, j, cur::jx2) * inv_sqrt_detH_ijP;
    m_mblock.em(i, j, em::ex3) += m_coeff * m_mblock.cur(i, j, cur::jx3) * inv_sqrt_detH_ij;
  }

  template <>
  Inline void
  AddCurrentsMinkowski<Dimension::THREE_D>::operator()(index_t i, index_t j, index_t k) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};
    real_t k_ {static_cast<real_t>(static_cast<int>(k) - N_GHOSTS)};
    real_t inv_sqrt_detH_iPjk {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_, k_})};
    real_t inv_sqrt_detH_ijPk {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF, k_})};
    real_t inv_sqrt_detH_ijkP {ONE / m_mblock.metric.sqrt_det_h({i_, j_, k_ + HALF})};
    m_mblock.em(i, j, k, em::ex1)
      += m_coeff * m_mblock.cur(i, j, k, cur::jx1) * inv_sqrt_detH_iPjk;
    m_mblock.em(i, j, k, em::ex2)
      += m_coeff * m_mblock.cur(i, j, k, cur::jx2) * inv_sqrt_detH_ijPk;
    m_mblock.em(i, j, k, em::ex3)
      += m_coeff * m_mblock.cur(i, j, k, cur::jx3) * inv_sqrt_detH_ijkP;
  }
} // namespace ntt

#endif
