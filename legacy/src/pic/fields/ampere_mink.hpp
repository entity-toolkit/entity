#ifndef PIC_AMPERE_MINKOWSKI_H
#define PIC_AMPERE_MINKOWSKI_H

#include "wrapper.h"

#include "field_macros.h"
#include "pic.h"

#include "io/output.h"
#include "meshblock/meshblock.h"

namespace ntt {
  /**
   * @brief Algorithm for the Ampere's law: `dE/dt = curl B` in Minkowski space.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Ampere_kernel {
    Meshblock<D, PICEngine> m_mblock;
    real_t                  m_coeff;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param coeff Coefficient to be multiplied by dE/dt = coeff * curl B.
     */
    Ampere_kernel(const Meshblock<D, PICEngine>& mblock, const real_t& coeff) :
      m_mblock(mblock),
      m_coeff(coeff) {}

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
  Inline void Ampere_kernel<Dim1>::operator()(index_t i) const {
    EX2(i) += m_coeff * (BX3(i - 1) - BX3(i));
    EX3(i) += m_coeff * (BX2(i) - BX2(i - 1));
  }

  template <>
  Inline void Ampere_kernel<Dim2>::operator()(index_t i, index_t j) const {
    EX1(i, j) += m_coeff * (BX3(i, j) - BX3(i, j - 1));
    EX2(i, j) += m_coeff * (BX3(i - 1, j) - BX3(i, j));
    EX3(i, j) += m_coeff * (BX1(i, j - 1) - BX1(i, j) + BX2(i, j) - BX2(i - 1, j));
  }

  template <>
  Inline void Ampere_kernel<Dim3>::operator()(index_t i, index_t j, index_t k) const {
    EX1(i, j, k) += m_coeff * (BX2(i, j, k - 1) - BX2(i, j, k) + BX3(i, j, k) -
                               BX3(i, j - 1, k));
    EX2(i, j, k) += m_coeff * (BX3(i - 1, j, k) - BX3(i, j, k) + BX1(i, j, k) -
                               BX1(i, j, k - 1));
    EX3(i, j, k) += m_coeff * (BX1(i, j - 1, k) - BX1(i, j, k) + BX2(i, j, k) -
                               BX2(i - 1, j, k));
  }
} // namespace ntt
#endif // PIC_AMPERE_MINKOWSKI_H