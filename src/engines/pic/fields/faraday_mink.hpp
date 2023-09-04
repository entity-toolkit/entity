#ifndef PIC_FARADAY_MINKOWSKI_H
#define PIC_FARADAY_MINKOWSKI_H

#include "wrapper.h"

#include "field_macros.h"
#include "pic.h"

#include "io/output.h"
#include "meshblock/meshblock.h"

namespace ntt {

  /**
   * @brief Algorithm for the Faraday's law: `dB/dt = -curl E` in Minkowski space.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Faraday_kernel {
    Meshblock<D, PICEngine> m_mblock;
    real_t                  m_coeff;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param coeff Coefficient to be multiplied by dB/dt = coeff * -curl E.
     */
    Faraday_kernel(const Meshblock<D, PICEngine>& mblock, const real_t& coeff) :
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
  Inline void Faraday_kernel<Dim1>::operator()(index_t i) const {
    BX2(i) += m_coeff * (EX3(i + 1) - EX3(i));
    BX3(i) += m_coeff * (EX2(i) - EX2(i + 1));
  }

  template <>
  Inline void Faraday_kernel<Dim2>::operator()(index_t i, index_t j) const {
    BX1(i, j) += m_coeff * (EX3(i, j) - EX3(i, j + 1));
    BX2(i, j) += m_coeff * (EX3(i + 1, j) - EX3(i, j));
    BX3(i, j) += m_coeff * (EX1(i, j + 1) - EX1(i, j) + EX2(i, j) - EX2(i + 1, j));
  }

  template <>
  Inline void Faraday_kernel<Dim3>::operator()(index_t i, index_t j, index_t k) const {
    BX1(i, j, k) += m_coeff * (EX2(i, j, k + 1) - EX2(i, j, k) + EX3(i, j, k) -
                               EX3(i, j + 1, k));
    BX2(i, j, k) += m_coeff * (EX3(i + 1, j, k) - EX3(i, j, k) + EX1(i, j, k) -
                               EX1(i, j, k + 1));
    BX3(i, j, k) += m_coeff * (EX1(i, j + 1, k) - EX1(i, j, k) + EX2(i, j, k) -
                               EX2(i + 1, j, k));
  }
} // namespace ntt

#endif
