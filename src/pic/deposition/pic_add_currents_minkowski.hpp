#ifndef PIC_ADD_CURRENTS_MINKOWSKI_H
#define PIC_ADD_CURRENTS_MINKOWSKI_H

#include "global.h"
#include "pic.h"

#include "field_macros.h"

namespace ntt {
  /**
   * @brief Add the currents to the E field (Minkowski).
   * @brief `m_coeff` includes metric coefficient.
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
    EX1(i) += m_coeff * JX1(i);
    EX2(i) += m_coeff * JX2(i);
    EX3(i) += m_coeff * JX3(i);
  }

  template <>
  Inline void AddCurrentsMinkowski<Dimension::TWO_D>::operator()(index_t i, index_t j) const {
    EX1(i, j) += m_coeff * JX1(i, j);
    EX2(i, j) += m_coeff * JX2(i, j);
    EX3(i, j) += m_coeff * JX3(i, j);
  }

  template <>
  Inline void
  AddCurrentsMinkowski<Dimension::THREE_D>::operator()(index_t i, index_t j, index_t k) const {
    EX1(i, j, k) += m_coeff * JX1(i, j, k);
    EX2(i, j, k) += m_coeff * JX2(i, j, k);
    EX3(i, j, k) += m_coeff * JX3(i, j, k);
  }
} // namespace ntt

#endif
