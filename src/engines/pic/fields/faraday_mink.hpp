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
    const float deltax = ZERO;
    const float alphax = ONE - THREE * deltax;
    
    BX2(i) += m_coeff * (
                      alphax * (EX3(i + 1)     - EX3(i    ))
                    + deltax * (EX3(i + 2)     - EX3(i - 1)));
    BX3(i) += m_coeff * (
                    - alphax * (EX2(i + 1)     - EX2(i    ))
                    - deltax * (EX2(i + 2)     - EX2(i - 1)));
  }

  template <>
  Inline void Faraday_kernel<Dim2>::operator()(index_t i, index_t j) const {
    // const float deltax = ZERO, deltay = ZERO,
    //             betaxy = ZERO, betayx = ZERO;
    const float deltax = -0.06500000000, deltay = -0.06500000000,
                betaxy = -0.06500000000, betayx = -0.06500000000;
    const float alphax = ONE - TWO * betaxy - THREE * deltax;
    const float alphay = ONE - TWO * betayx - THREE * deltay;
    
    BX1(i, j) += m_coeff * (
                    - alphay * (EX3(i    , j + 1) - EX3(i    , j    ))
                    - deltay * (EX3(i    , j + 2) - EX3(i    , j - 1))
                    - betayx * (EX3(i + 1, j + 1) - EX3(i + 1, j    ))
                    - betayx * (EX3(i - 1, j + 1) - EX3(i - 1, j    )));
    BX2(i, j) += m_coeff * (
                      alphax * (EX3(i + 1, j    ) - EX3(i    , j    ))
                    + deltax * (EX3(i + 2, j    ) - EX3(i - 1, j    ))
                    + betaxy * (EX3(i + 1, j + 1) - EX3(i    , j + 1))
                    + betaxy * (EX3(i + 1, j - 1) - EX3(i    , j - 1)));
    BX3(i, j) += m_coeff * (
                      alphay * (EX1(i    , j + 1) - EX1(i    , j    ))
                    + deltay * (EX1(i    , j + 2) - EX1(i    , j - 1))
                    + betayx * (EX1(i + 1, j + 1) - EX1(i + 1, j    ))
                    + betayx * (EX1(i - 1, j + 1) - EX1(i - 1, j    ))
                    - alphax * (EX2(i + 1, j    ) - EX2(i    , j    ))
                    - deltax * (EX2(i + 2, j    ) - EX2(i - 1, j    ))
                    - betaxy * (EX2(i + 1, j + 1) - EX2(i    , j + 1))
                    - betaxy * (EX2(i + 1, j - 1) - EX2(i    , j - 1)));
  }

  template <>
  Inline void Faraday_kernel<Dim3>::operator()(index_t i, index_t j, index_t k) const {
    const float deltax = ZERO, deltay = ZERO, deltaz = ZERO,
                betaxy = ZERO, betayx = ZERO, betaxz = ZERO,
                betazx = ZERO, betayz = ZERO, betazy = ZERO;
    const float alphax = ONE - TWO * betaxy - TWO * betaxz - THREE * deltax;
    const float alphay = ONE - TWO * betayx - TWO * betayz - THREE * deltay;
    const float alphaz = ONE - TWO * betazx - TWO * betazy - THREE * deltaz;

    BX1(i, j, k) += m_coeff * (
                      alphaz * (EX2(i    , j    , k + 1) - EX2(i    , j    , k    ))
                    + deltaz * (EX2(i    , j    , k + 2) - EX2(i    , j    , k - 1))
                    + betazx * (EX2(i + 1, j    , k + 1) - EX2(i + 1, j    , k    ))
                    + betazx * (EX2(i - 1, j    , k + 1) - EX2(i - 1, j    , k    ))
                    + betazy * (EX2(i    , j + 1, k + 1) - EX2(i    , j + 1, k    ))
                    + betazy * (EX2(i    , j - 1, k + 1) - EX2(i    , j - 1, k    ))
                    - alphay * (EX3(i    , j + 1, k    ) - EX3(i    , j    , k    ))
                    - deltay * (EX3(i    , j + 2, k    ) - EX3(i    , j - 1, k    ))
                    - betayx * (EX3(i + 1, j + 1, k    ) - EX3(i + 1, j    , k    ))
                    - betayx * (EX3(i - 1, j + 1, k    ) - EX3(i - 1, j    , k    ))
                    - betayz * (EX3(i    , j + 1, k + 1) - EX3(i    , j    , k + 1))
                    - betayz * (EX3(i    , j + 1, k - 1) - EX3(i    , j    , k - 1)));
    BX2(i, j, k) += m_coeff * (
                      alphax * (EX3(i + 1, j    , k    ) - EX3(i    , j    , k    ))
                    + deltax * (EX3(i + 2, j    , k    ) - EX3(i - 1, j    , k    ))
                    + betaxy * (EX3(i + 1, j + 1, k    ) - EX3(i    , j + 1, k    ))
                    + betaxy * (EX3(i + 1, j - 1, k    ) - EX3(i    , j - 1, k    ))
                    + betaxz * (EX3(i + 1, j    , k + 1) - EX3(i    , j    , k + 1))
                    + betaxz * (EX3(i + 1, j    , k - 1) - EX3(i    , j    , k - 1))
                    - alphaz * (EX1(i    , j    , k + 1) - EX1(i    , j    , k    ))
                    - deltaz * (EX1(i    , j    , k + 2) - EX1(i    , j    , k - 1))
                    - betazx * (EX1(i + 1, j    , k + 1) - EX1(i + 1, j    , k    ))
                    - betazx * (EX1(i - 1, j    , k + 1) - EX1(i - 1, j    , k    ))
                    - betazy * (EX1(i    , j + 1, k + 1) - EX1(i    , j + 1, k    ))
                    - betazy * (EX1(i    , j - 1, k + 1) - EX1(i    , j - 1, k    )));
    BX3(i, j, k) += m_coeff * (
                      alphay * (EX1(i    , j + 1, k    ) - EX1(i    , j    , k    ))
                    + deltay * (EX1(i    , j + 2, k    ) - EX1(i    , j - 1, k    ))
                    + betayx * (EX1(i + 1, j + 1, k    ) - EX1(i + 1, j    , k    ))
                    + betayx * (EX1(i - 1, j + 1, k    ) - EX1(i - 1, j    , k    ))
                    + betayz * (EX1(i    , j + 1, k + 1) - EX1(i    , j    , k + 1))
                    + betayz * (EX1(i    , j + 1, k - 1) - EX1(i    , j    , k - 1))
                    - alphax * (EX2(i + 1, j    , k    ) - EX2(i    , j    , k    ))
                    - deltax * (EX2(i + 2, j    , k    ) - EX2(i - 1, j    , k    ))
                    - betaxy * (EX2(i + 1, j + 1, k    ) - EX2(i    , j + 1, k    ))
                    - betaxy * (EX2(i + 1, j - 1, k    ) - EX2(i    , j - 1, k     ))
                    - betaxz * (EX2(i + 1, j    , k + 1) - EX2(i    , j    , k + 1))
                    - betaxz * (EX2(i + 1, j    , k - 1) - EX2(i    , j    , k - 1)));
  }
} // namespace ntt

#endif
