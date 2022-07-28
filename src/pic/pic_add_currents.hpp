#ifndef PIC_ADD_CURRENTS_H
#define PIC_ADD_CURRENTS_H

#include "global.h"
#include "pic.h"

namespace ntt {
  /**
   * @brief Add the currents to the E field.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class AddCurrentsSubstep {
    Meshblock<D, SimulationType::PIC> m_mblock;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     */
    AddCurrentsSubstep(const Meshblock<D, SimulationType::PIC>& mblock) : m_mblock {mblock} {}
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
  Inline void AddCurrentsSubstep<Dimension::ONE_D>::operator()(index_t i) const {
    m_mblock.em(i, em::ex1) += m_mblock.cur(i, cur::jx1);
    m_mblock.em(i, em::ex2) += m_mblock.cur(i, cur::jx2);
    m_mblock.em(i, em::ex3) += m_mblock.cur(i, cur::jx3);
  }

  template <>
  Inline void AddCurrentsSubstep<Dimension::TWO_D>::operator()(index_t i,
                                                               index_t j) const {
    m_mblock.em(i, j, em::ex1) += m_mblock.cur(i, j, cur::jx1);
    m_mblock.em(i, j, em::ex2) += m_mblock.cur(i, j, cur::jx2);
    m_mblock.em(i, j, em::ex3) += m_mblock.cur(i, j, cur::jx3);
  }

  template <>
  Inline void AddCurrentsSubstep<Dimension::THREE_D>::operator()(index_t i,
                                                                 index_t j,
                                                                 index_t k) const {
    m_mblock.em(i, j, k, em::ex1) += m_mblock.cur(i, j, k, cur::jx1);
    m_mblock.em(i, j, k, em::ex2) += m_mblock.cur(i, j, k, cur::jx2);
    m_mblock.em(i, j, k, em::ex3) += m_mblock.cur(i, j, k, cur::jx3);
  }
} // namespace ntt

#endif
