#ifndef PIC_FARADAY_MINKOWSKI_H
#define PIC_FARADAY_MINKOWSKI_H

#include "global.h"
#include "fields.h"
#include "meshblock.h"
#include "pic.h"

namespace ntt {

  /**
   * @brief Algorithm for the Faraday's law: `dB/dt = -curl E` in Minkowski space.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class FaradayMinkowski {
    Meshblock<D, SimulationType::PIC> m_mblock;
    real_t                            m_coeff;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param coeff Coefficient to be multiplied by dB/dt = coeff * -curl E.
     */
    FaradayMinkowski(const Meshblock<D, SimulationType::PIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff) {}
    /**
     * @brief 1D implementation of the algorithm.
     * @param i1 index.
     */
    Inline void operator()(index_t) const;
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
  Inline void FaradayMinkowski<Dimension::ONE_D>::operator()(index_t i) const {
    m_mblock.em(i, em::bx2)
      += m_coeff * (m_mblock.em(i + 1, em::ex3) - m_mblock.em(i, em::ex3));
    m_mblock.em(i, em::bx3)
      += m_coeff * (m_mblock.em(i, em::ex2) - m_mblock.em(i + 1, em::ex2));
  }

  template <>
  Inline void FaradayMinkowski<Dimension::TWO_D>::operator()(index_t i, index_t j) const {
    m_mblock.em(i, j, em::bx1)
      += m_coeff * (m_mblock.em(i, j, em::ex3) - m_mblock.em(i, j + 1, em::ex3));
    m_mblock.em(i, j, em::bx2)
      += m_coeff * (m_mblock.em(i + 1, j, em::ex3) - m_mblock.em(i, j, em::ex3));
    m_mblock.em(i, j, em::bx3)
      += m_coeff
         * (m_mblock.em(i, j + 1, em::ex1) - m_mblock.em(i, j, em::ex1)
            + m_mblock.em(i, j, em::ex2) - m_mblock.em(i + 1, j, em::ex2));
  }

  template <>
  Inline void
  FaradayMinkowski<Dimension::THREE_D>::operator()(index_t i, index_t j, index_t k) const {
    m_mblock.em(i, j, k, em::bx1)
      += m_coeff
         * (m_mblock.em(i, j, k + 1, em::ex2) - m_mblock.em(i, j, k, em::ex2)
            + m_mblock.em(i, j, k, em::ex3) - m_mblock.em(i, j + 1, k, em::ex3));
    m_mblock.em(i, j, k, em::bx2)
      += m_coeff
         * (m_mblock.em(i + 1, j, k, em::ex3) - m_mblock.em(i, j, k, em::ex3)
            + m_mblock.em(i, j, k, em::ex1) - m_mblock.em(i, j, k + 1, em::ex1));
    m_mblock.em(i, j, k, em::bx3)
      += m_coeff
         * (m_mblock.em(i, j + 1, k, em::ex1) - m_mblock.em(i, j, k, em::ex1)
            + m_mblock.em(i, j, k, em::ex2) - m_mblock.em(i + 1, j, k, em::ex2));
  }
} // namespace ntt

#endif
