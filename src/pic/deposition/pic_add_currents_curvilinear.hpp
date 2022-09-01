#ifndef PIC_ADD_CURRENTS_MINKOWSKI_H
#define PIC_ADD_CURRENTS_MINKOWSKI_H

#include "global.h"
#include "pic.h"

#include "field_macros.h"

// !TODO: Wrong treatment of the axes

namespace ntt {
  /**
   * @brief Add the currents to the E field with the appropriate conversion.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class AddCurrentsCurvilinear {
    Meshblock<D, SimulationType::PIC> m_mblock;
    real_t                            m_coeff;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     */
    AddCurrentsCurvilinear(const Meshblock<D, SimulationType::PIC>& mblock,
                           const real_t&                            coeff)
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
  Inline void AddCurrentsCurvilinear<Dim1>::operator()(index_t i) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t inv_sqrt_detH_i {ONE / m_mblock.metric.sqrt_det_h({i_})};
    real_t inv_sqrt_detH_iP {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF})};
    EX1(i) += m_coeff * JX1(i) * inv_sqrt_detH_iP;
    EX2(i) += m_coeff * JX2(i) * inv_sqrt_detH_i;
    EX3(i) += m_coeff * JX3(i) * inv_sqrt_detH_i;
  }

  template <>
  Inline void AddCurrentsCurvilinear<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};
    real_t inv_sqrt_detH_ij {ONE / m_mblock.metric.sqrt_det_h({i_, j_})};
    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    EX1(i, j) += m_coeff * JX1(i, j) * inv_sqrt_detH_iPj;
    EX2(i, j) += m_coeff * JX2(i, j) * inv_sqrt_detH_ijP;
    EX3(i, j) += m_coeff * JX3(i, j) * inv_sqrt_detH_ij;
  }

  template <>
  Inline void AddCurrentsCurvilinear<Dim3>::operator()(index_t i, index_t j, index_t k) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};
    real_t k_ {static_cast<real_t>(static_cast<int>(k) - N_GHOSTS)};
    real_t inv_sqrt_detH_iPjk {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_, k_})};
    real_t inv_sqrt_detH_ijPk {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF, k_})};
    real_t inv_sqrt_detH_ijkP {ONE / m_mblock.metric.sqrt_det_h({i_, j_, k_ + HALF})};
    EX1(i, j, k) += m_coeff * JX1(i, j, k) * inv_sqrt_detH_iPjk;
    EX2(i, j, k) += m_coeff * JX2(i, j, k) * inv_sqrt_detH_ijPk;
    EX3(i, j, k) += m_coeff * JX3(i, j, k) * inv_sqrt_detH_ijkP;
  }
} // namespace ntt

#endif
