#ifndef PIC_ADD_CURRENTS_H
#define PIC_ADD_CURRENTS_H

#include "global.h"
#include "pic.h"

namespace ntt {
  /**
   * Add the currents to the E field.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class AddCurrentsSubstep {
    using index_t = typename RealFieldND<D, 3>::size_type;
    Meshblock<D, SimulationType::PIC> m_mblock;

  public:
    AddCurrentsSubstep(const Meshblock<D, SimulationType::PIC>& mblock) : m_mblock {mblock} {}
    Inline void operator()(const index_t) const;
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  template <>
  Inline void AddCurrentsSubstep<Dimension::ONE_D>::operator()(const index_t i) const {
    m_mblock.em(i, em::ex1) += m_mblock.cur(i, cur::jx1);
    m_mblock.em(i, em::ex2) += m_mblock.cur(i, cur::jx2);
    m_mblock.em(i, em::ex3) += m_mblock.cur(i, cur::jx3);
  }

  template <>
  Inline void AddCurrentsSubstep<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {
    //m_mblock.em(i, j, em::ex1) += m_mblock.cur(i, j, cur::jx1);
    //m_mblock.em(i, j, em::ex2) += m_mblock.cur(i, j, cur::jx2);
    m_mblock.em(i, j, em::ex3) += m_mblock.cur(i, j, cur::jx3);
  }

  template <>
  Inline void AddCurrentsSubstep<Dimension::THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
    m_mblock.em(i, j, k, em::ex1) += m_mblock.cur(i, j, k, cur::jx1);
    m_mblock.em(i, j, k, em::ex2) += m_mblock.cur(i, j, k, cur::jx2);
    m_mblock.em(i, j, k, em::ex3) += m_mblock.cur(i, j, k, cur::jx3);
  }
} // namespace ntt

#endif
