#ifndef GRPIC_CURRENTS_AMPERE_H
#define GRPIC_CURRENTS_AMPERE_H

#include "wrapper.h"

#include "field_macros.h"
#include "grpic.h"

namespace ntt {
  /**
   * @brief Add the currents to the D field with the appropriate conversion.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class CurrentsAmpere_kernel {
    Meshblock<D, GRPICEngine> m_mblock;
    real_t                    m_coeff;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     */
    CurrentsAmpere_kernel(const Meshblock<D, GRPICEngine>& mblock, const real_t& coeff)
      : m_mblock { mblock }, m_coeff { coeff } {}
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
  Inline void CurrentsAmpere_kernel<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };
    real_t inv_sqrt_detH_ij { ONE / m_mblock.metric.sqrt_det_h({ i_, j_ }) };
    real_t inv_sqrt_detH_iPj { ONE / m_mblock.metric.sqrt_det_h({ i_ + HALF, j_ }) };
    real_t inv_sqrt_detH_ijP { ONE / m_mblock.metric.sqrt_det_h({ i_, j_ + HALF }) };

    J0X1(i, j) *= m_coeff * inv_sqrt_detH_iPj;
    J0X2(i, j) *= m_coeff * inv_sqrt_detH_ijP;
    J0X3(i, j) *= m_coeff * inv_sqrt_detH_ij;

    D0X1(i, j) += J0X1(i, j);
    D0X2(i, j) += J0X2(i, j);
    D0X3(i, j) += J0X3(i, j);
  }

  template <>
  Inline void CurrentsAmpere_kernel<Dim3>::operator()(index_t i, index_t j, index_t k) const {
    real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };
    real_t k_ { static_cast<real_t>(static_cast<int>(k) - N_GHOSTS) };
    real_t inv_sqrt_detH_iPjk { ONE / m_mblock.metric.sqrt_det_h({ i_ + HALF, j_, k_ }) };
    real_t inv_sqrt_detH_ijPk { ONE / m_mblock.metric.sqrt_det_h({ i_, j_ + HALF, k_ }) };
    real_t inv_sqrt_detH_ijkP { ONE / m_mblock.metric.sqrt_det_h({ i_, j_, k_ + HALF }) };

    J0X1(i, j, k) *= m_coeff * inv_sqrt_detH_iPjk;
    J0X2(i, j, k) *= m_coeff * inv_sqrt_detH_ijPk;
    J0X3(i, j, k) *= m_coeff * inv_sqrt_detH_ijkP;

    D0X1(i, j, k) += J0X1(i, j, k);
    D0X2(i, j, k) += J0X2(i, j, k);
    D0X3(i, j, k) += J0X3(i, j, k);
  }

  template <Dimension D>
  class CurrentsAmperePoles_kernel {
    Meshblock<D, GRPICEngine> m_mblock;
    real_t                    m_coeff;
    const std::size_t         m_ni2;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     */
    CurrentsAmperePoles_kernel(const Meshblock<D, GRPICEngine>& mblock, const real_t& coeff)
      : m_mblock { mblock }, m_coeff { coeff }, m_ni2 { m_mblock.Ni2() } {}
    /**
     * @param i index.
     */
    Inline void operator()(index_t i) const;
  };

  template <>
  Inline void CurrentsAmperePoles_kernel<Dim2>::operator()(index_t i) const {
    index_t j_min { N_GHOSTS };
    index_t j_max { m_ni2 + N_GHOSTS };

    real_t  i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };

    real_t  inv_sqrt_detH_ijP { ONE / m_mblock.metric.sqrt_det_h({ i_, HALF }) };
    real_t  inv_polar_area_iPj { ONE / m_mblock.metric.polar_area(i_ + HALF) };
    // theta = 0
    J0X1(i, j_min) *= HALF * m_coeff * inv_polar_area_iPj;
    D0X1(i, j_min) += J0X1(i, j_min);

    // theta = pi
    J0X1(i, j_max) *= HALF * m_coeff * inv_polar_area_iPj;
    D0X1(i, j_max) += J0X1(i, j_max);

    // j = jmin + 1/2
    J0X2(i, j_min) *= m_coeff * inv_sqrt_detH_ijP;
    D0X2(i, j_min) += J0X2(i, j_min);
  }

  /**
   * @brief Add the currents to the D field with the appropriate conversion.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class CurrentsAmpereAux_kernel {
    Meshblock<D, GRPICEngine> m_mblock;
    real_t                    m_coeff;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     */
    CurrentsAmpereAux_kernel(const Meshblock<D, GRPICEngine>& mblock, const real_t& coeff)
      : m_mblock { mblock }, m_coeff { coeff } {}
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
  Inline void CurrentsAmpereAux_kernel<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };
    real_t inv_sqrt_detH_ij { ONE / m_mblock.metric.sqrt_det_h({ i_, j_ }) };
    real_t inv_sqrt_detH_iPj { ONE / m_mblock.metric.sqrt_det_h({ i_ + HALF, j_ }) };
    real_t inv_sqrt_detH_ijP { ONE / m_mblock.metric.sqrt_det_h({ i_, j_ + HALF }) };

    JX1(i, j) *= m_coeff * inv_sqrt_detH_iPj;
    JX2(i, j) *= m_coeff * inv_sqrt_detH_ijP;
    JX3(i, j) *= m_coeff * inv_sqrt_detH_ij;

    D0X1(i, j) += JX1(i, j);
    D0X2(i, j) += JX2(i, j);
    D0X3(i, j) += JX3(i, j);
  }

  template <>
  Inline void CurrentsAmpereAux_kernel<Dim3>::operator()(index_t i,
                                                         index_t j,
                                                         index_t k) const {
    real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };
    real_t k_ { static_cast<real_t>(static_cast<int>(k) - N_GHOSTS) };
    real_t inv_sqrt_detH_iPjk { ONE / m_mblock.metric.sqrt_det_h({ i_ + HALF, j_, k_ }) };
    real_t inv_sqrt_detH_ijPk { ONE / m_mblock.metric.sqrt_det_h({ i_, j_ + HALF, k_ }) };
    real_t inv_sqrt_detH_ijkP { ONE / m_mblock.metric.sqrt_det_h({ i_, j_, k_ + HALF }) };

    JX1(i, j, k) *= m_coeff * inv_sqrt_detH_iPjk;
    JX2(i, j, k) *= m_coeff * inv_sqrt_detH_ijPk;
    JX3(i, j, k) *= m_coeff * inv_sqrt_detH_ijkP;

    D0X1(i, j, k) += JX1(i, j, k);
    D0X2(i, j, k) += JX2(i, j, k);
    D0X3(i, j, k) += JX3(i, j, k);
  }

  template <Dimension D>
  class CurrentsAmpereAuxPoles_kernel {
    Meshblock<D, GRPICEngine> m_mblock;
    real_t                    m_coeff;
    const std::size_t         m_ni2;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     */
    CurrentsAmpereAuxPoles_kernel(const Meshblock<D, GRPICEngine>& mblock, const real_t& coeff)
      : m_mblock { mblock }, m_coeff { coeff }, m_ni2 { m_mblock.Ni2() } {}
    /**
     * @param i index.
     */
    Inline void operator()(index_t i) const;
  };

  template <>
  Inline void CurrentsAmpereAuxPoles_kernel<Dim2>::operator()(index_t i) const {
    index_t j_min { N_GHOSTS };
    index_t j_max { m_ni2 + N_GHOSTS };

    real_t  i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };

    real_t  inv_sqrt_detH_ijP { ONE / m_mblock.metric.sqrt_det_h({ i_, HALF }) };
    real_t  inv_polar_area_iPj { ONE / m_mblock.metric.polar_area(i_ + HALF) };

    // theta = 0
    JX1(i, j_min) *= HALF * m_coeff * inv_polar_area_iPj;
    D0X1(i, j_min) += JX1(i, j_min);

    // theta = pi
    JX1(i, j_max) *= HALF * m_coeff * inv_polar_area_iPj;
    D0X1(i, j_max) += JX1(i, j_max);

    // j = jmin + 1/2
    JX2(i, j_min) *= m_coeff * inv_sqrt_detH_ijP;
    D0X2(i, j_min) += JX2(i, j_min);
  }
}    // namespace ntt

#endif    // GRPIC_CURRENTS_AMPERE_H
