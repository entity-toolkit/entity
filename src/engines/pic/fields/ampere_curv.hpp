#ifndef PIC_AMPERE_CURVILINEAR_H
#define PIC_AMPERE_CURVILINEAR_H

#include "wrapper.h"

#include "field_macros.h"
#include "pic.h"

#include "io/output.h"
#include "meshblock/meshblock.h"

namespace ntt {
  /**
   * @brief Algorithm for the Ampere's law: `dE/dt = curl B` in curvilinear space.
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
     * @brief 2D version of the algorithm.
     * @param i1 index.
     * @param i2 index.
     */
    Inline void operator()(index_t i1, index_t i2) const;
    /**
     * @brief 3D version of the algorithm.
     * @param i1 index.
     * @param i2 index.
     * @param i3 index.
     */
    Inline void operator()(index_t i1, index_t i2, index_t i3) const;
  };

  template <>
  Inline void Ampere_kernel<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

    real_t inv_sqrt_detH_ij { ONE / m_mblock.metric.sqrt_det_h({ i_, j_ }) };
    real_t inv_sqrt_detH_iPj { ONE / m_mblock.metric.sqrt_det_h({ i_ + HALF, j_ }) };
    real_t inv_sqrt_detH_ijP { ONE / m_mblock.metric.sqrt_det_h({ i_, j_ + HALF }) };
    real_t h1_ijM { m_mblock.metric.h_11({ i_, j_ - HALF }) };
    real_t h1_ijP { m_mblock.metric.h_11({ i_, j_ + HALF }) };
    real_t h2_iPj { m_mblock.metric.h_22({ i_ + HALF, j_ }) };
    real_t h2_iMj { m_mblock.metric.h_22({ i_ - HALF, j_ }) };
    real_t h3_iMjP { m_mblock.metric.h_33({ i_ - HALF, j_ + HALF }) };
    real_t h3_iPjM { m_mblock.metric.h_33({ i_ + HALF, j_ - HALF }) };
    real_t h3_iPjP { m_mblock.metric.h_33({ i_ + HALF, j_ + HALF }) };

    EX1(i, j) += m_coeff * inv_sqrt_detH_iPj *
                 (h3_iPjP * BX3(i, j) - h3_iPjM * BX3(i, j - 1));
    EX2(i, j) += m_coeff * inv_sqrt_detH_ijP *
                 (h3_iMjP * BX3(i - 1, j) - h3_iPjP * BX3(i, j));
    EX3(i, j) += m_coeff * inv_sqrt_detH_ij *
                 (h1_ijM * BX1(i, j - 1) - h1_ijP * BX1(i, j) +
                  h2_iPj * BX2(i, j) - h2_iMj * BX2(i - 1, j));
  }

  template <>
  Inline void Ampere_kernel<Dim3>::operator()(index_t, index_t, index_t) const {
    // 3d curvilinear ampere not implemented
  }

  /**
   * @brief Algorithm for the Ampere's law: `dE/dt = curl B` in curvilinear
   * space near the polar axes (integral form).
   * @tparam D Dimension.
   */
  template <Dimension D>
  class AmperePoles_kernel {
    Meshblock<D, PICEngine> m_mblock;
    real_t                  m_coeff;
    const std::size_t       m_ni2;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param coeff Coefficient to be multiplied by dE/dt = coeff * curl B.
     */
    AmperePoles_kernel(const Meshblock<D, PICEngine>& mblock, const real_t& coeff) :
      m_mblock(mblock),
      m_coeff(coeff),
      m_ni2(m_mblock.Ni2()) {}

    /**
     * @brief Implementation of the algorithm.
     * @param i radial index.
     */
    Inline void operator()(index_t i) const;
  };

  template <>
  Inline void AmperePoles_kernel<Dim2>::operator()(index_t i) const {
    index_t j_min { N_GHOSTS };
    index_t j_max { m_ni2 + N_GHOSTS };

    real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    real_t j_max_ { static_cast<real_t>(static_cast<int>(j_max) - N_GHOSTS) };

    real_t inv_polar_area_iPj { ONE / m_mblock.metric.polar_area(i_ + HALF) };
    real_t h3_min_iPjP { m_mblock.metric.h_33({ i_ + HALF, HALF }) };
    real_t h3_max_iPjM { m_mblock.metric.h_33({ i_ + HALF, j_max_ - HALF }) };

    real_t inv_sqrt_detH_ijP { ONE / m_mblock.metric.sqrt_det_h({ i_, HALF }) };
    real_t h3_min_iMjP { m_mblock.metric.h_33({ i_ - HALF, HALF }) };

    // theta = 0
    EX1(i, j_min) += inv_polar_area_iPj * m_coeff * (h3_min_iPjP * BX3(i, j_min));
    // theta = pi
    EX1(i, j_max) -= inv_polar_area_iPj * m_coeff * (h3_max_iPjM * BX3(i, j_max));

    // j = jmin + 1/2
    EX2(i, j_min) += inv_sqrt_detH_ijP * m_coeff *
                     (h3_min_iMjP * BX3(i - 1, j_min) -
                      h3_min_iPjP * BX3(i, j_min));
  }
} // namespace ntt

#endif // NTT_AMPERE_KERNEL_HPP
