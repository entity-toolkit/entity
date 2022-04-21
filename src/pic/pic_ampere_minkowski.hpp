#ifndef PIC_AMPERE_MINKOWSKI_H
#define PIC_AMPERE_MINKOWSKI_H

#include "global.h"
#include "fields.h"
#include "meshblock.h"
#include "pic.h"

namespace ntt {

  /**
   * Algorithm for the Ampere's law: `dE/dt = curl B` in Minkowski space.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class AmpereMinkowski {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::PIC> m_mblock;
    real_t m_coeff;

  public:
    AmpereMinkowski(const Meshblock<D, SimulationType::PIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff) {}
    Inline void operator()(const index_t) const;
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  template <>
  Inline void AmpereMinkowski<Dimension::ONE_D>::operator()(const index_t i) const {
    m_mblock.em(i, em::ex2) += m_coeff * (
                                          m_mblock.em(i - 1, em::bx3) - m_mblock.em(i, em::bx3)
                                        );
    m_mblock.em(i, em::ex3) += m_coeff * (
                                          m_mblock.em(i, em::bx2) - m_mblock.em(i - 1, em::bx2)
                                        );
  }

  template <>
  Inline void AmpereMinkowski<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {
    m_mblock.em(i, j, em::ex1) += m_coeff * (
                                            m_mblock.em(i, j, em::bx3) - m_mblock.em(i, j - 1, em::bx3)
                                          );
    m_mblock.em(i, j, em::ex2) += m_coeff * (
                                            m_mblock.em(i - 1, j, em::bx3) - m_mblock.em(i, j, em::bx3)
                                          );
    m_mblock.em(i, j, em::ex3) += m_coeff * (
                                            m_mblock.em(i, j - 1, em::bx1) - m_mblock.em(i, j, em::bx1) +
                                            m_mblock.em(i, j, em::bx2) - m_mblock.em(i - 1, j, em::bx2)
                                          );
  }

  template <>
  Inline void AmpereMinkowski<Dimension::THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
    m_mblock.em(i, j, k, em::ex1) += m_coeff * (
                                                m_mblock.em(i, j, k - 1, em::bx2) - m_mblock.em(i, j, k, em::bx2) +
                                                m_mblock.em(i, j, k, em::bx3) - m_mblock.em(i, j - 1, k, em::bx3)
                                              );
    m_mblock.em(i, j, k, em::ex2) += m_coeff * (
                                                m_mblock.em(i - 1, j, k, em::bx3) - m_mblock.em(i, j, k, em::bx3) +
                                                m_mblock.em(i, j, k, em::bx1) - m_mblock.em(i, j, k - 1, em::bx1)
                                              );
    m_mblock.em(i, j, k, em::ex3) += m_coeff * (
                                                m_mblock.em(i, j - 1, k, em::bx1) - m_mblock.em(i, j, k, em::bx1) +
                                                m_mblock.em(i, j, k, em::bx2) - m_mblock.em(i - 1, j, k, em::bx2)
                                              );
  }
} // namespace ntt

#endif
