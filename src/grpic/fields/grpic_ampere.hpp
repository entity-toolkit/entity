#ifndef GRPIC_AMPERE_H
#define GRPIC_AMPERE_H

#include "global.h"
#include "fields.h"
#include "meshblock.h"
#include "grpic.h"
#include "field_macros.h"

namespace ntt {

  /**
   * Algorithms for Ampere's law: `dD/dt = curl H`.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class AmpereGR_aux {

    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t                              m_coeff;

  public:
    AmpereGR_aux(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff) {}
    Inline void operator()(index_t, index_t) const;
    Inline void operator()(index_t, index_t, index_t) const;
  };

  // First push, updates D0 with J.
  template <>
  Inline void AmpereGR_aux<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};

    real_t inv_sqrt_detH_ij {ONE / m_mblock.metric.sqrt_det_h({i_, j_})};
    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};

    D0X1(i, j) += m_coeff * inv_sqrt_detH_iPj * (HX3(i, j) - HX3(i, j - 1));
    D0X2(i, j) += m_coeff * inv_sqrt_detH_ijP * (HX3(i - 1, j) - HX3(i, j));
    D0X3(i, j)
      += m_coeff * inv_sqrt_detH_ij * (HX1(i, j - 1) - HX1(i, j) + HX2(i, j) - HX2(i - 1, j));
  }

  template <>
  Inline void AmpereGR_aux<Dim3>::operator()(index_t, index_t, index_t) const {
    // 3d curvilinear ampere not implemented
  }

  template <Dimension D>
  class AmpereGR {

    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t                              m_coeff;

  public:
    AmpereGR(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff) {}
    Inline void operator()(index_t, index_t) const;
    Inline void operator()(index_t, index_t, index_t) const;
  };

  // Second push, updates D with J0 but assigns it to D0.
  template <>
  Inline void AmpereGR<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};

    real_t inv_sqrt_detH_ij {ONE / m_mblock.metric.sqrt_det_h({i_, j_})};
    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};

    D0X1(i, j) = DX1(i, j) + m_coeff * inv_sqrt_detH_iPj * (HX3(i, j) - HX3(i, j - 1));
    D0X2(i, j) = DX2(i, j) + m_coeff * inv_sqrt_detH_ijP * (HX3(i - 1, j) - HX3(i, j));
    D0X3(i, j)
      = DX3(i, j)
        + m_coeff * inv_sqrt_detH_ij * (HX1(i, j - 1) - HX1(i, j) + HX2(i, j) - HX2(i - 1, j));
  }

  template <>
  Inline void AmpereGR<Dim3>::operator()(index_t, index_t, index_t) const {
    // 3d curvilinear ampere not implemented
  }

  template <Dimension D>
  class AmpereGR_init {

    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t                              m_coeff;

  public:
    AmpereGR_init(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff) {}
    Inline void operator()(index_t, index_t) const;
    Inline void operator()(index_t, index_t, index_t) const;
  };

  // Second push, updates D with J0 but assigns it to D0.
  template <>
  Inline void AmpereGR_init<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};

    real_t inv_sqrt_detH_ij {ONE / m_mblock.metric.sqrt_det_h({i_, j_})};
    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};

    DX1(i, j) += m_coeff * inv_sqrt_detH_iPj * (HX3(i, j) - HX3(i, j - 1));
    DX2(i, j) += m_coeff * inv_sqrt_detH_ijP * (HX3(i - 1, j) - HX3(i, j));
    DX3(i, j)
      += m_coeff * inv_sqrt_detH_ij * (HX1(i, j - 1) - HX1(i, j) + HX2(i, j) - HX2(i - 1, j));
  }

  template <>
  Inline void AmpereGR_init<Dim3>::operator()(index_t, index_t, index_t) const {
    NTTError("3D GRPIC not implemented yet");
  }

  /**
   * Algorithms for Ampere's law: `dD/dt = curl H` near the polar axes (integral form).
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class AmperePolesGR_aux {

    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t                              m_coeff;
    const std::size_t                   m_nj;

  public:
    AmperePolesGR_aux(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff), m_nj(m_mblock.Ni2()) {}
    Inline void operator()(index_t) const;
  };

  // First push, updates D0 with J.
  template <>
  Inline void AmperePolesGR_aux<Dim2>::operator()(index_t i) const {
    index_t j_min {N_GHOSTS};
    index_t j_max {m_nj + N_GHOSTS - 1};
    real_t  i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};

    real_t inv_polar_area_iPj {ONE / m_mblock.metric.polar_area({i_ + HALF, HALF})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, HALF})};

    // theta = 0
    D0X1(i, j_min) += inv_polar_area_iPj * m_coeff * HX3(i, j_min);
    // theta = pi
    D0X1(i, j_max + 1) -= inv_polar_area_iPj * m_coeff * HX3(i, j_max);
    // j = jmin + 1/2
    D0X2(i, j_min) += inv_sqrt_detH_ijP * m_coeff * (HX3(i - 1, j_min) - HX3(i, j_min));
  }

  template <Dimension D>
  class AmperePolesGR {

    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t                              m_coeff;
    const std::size_t                   m_nj;

  public:
    AmperePolesGR(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff), m_nj(m_mblock.Ni2()) {}
    Inline void operator()(index_t) const;
  };

  // Second push, updates D with J0 but assigns it to D0.
  template <>
  Inline void AmperePolesGR<Dim2>::operator()(index_t i) const {
    index_t j_min {N_GHOSTS};
    index_t j_max {m_nj + N_GHOSTS - 1};
    real_t  i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};

    real_t inv_polar_area_iPj {ONE / m_mblock.metric.polar_area({i_ + HALF, HALF})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, HALF})};

    // theta = 0
    D0X1(i, j_min) = DX1(i, j_min) + inv_polar_area_iPj * m_coeff * HX3(i, j_min);
    // theta = pi
    D0X1(i, j_max + 1) = DX1(i, j_max + 1) - inv_polar_area_iPj * m_coeff * HX3(i, j_max);
    // j = jmin + 1/2
    D0X2(i, j_min)
      = DX2(i, j_min) + inv_sqrt_detH_ijP * m_coeff * (HX3(i - 1, j_min) - HX3(i, j_min));
  }

  template <Dimension D>
  class AmperePolesGR_init {

    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t                              m_coeff;
    const std::size_t                   m_nj;

  public:
    AmperePolesGR_init(const Meshblock<D, SimulationType::GRPIC>& mblock, const real_t& coeff)
      : m_mblock(mblock), m_coeff(coeff), m_nj(m_mblock.Ni2()) {}
    Inline void operator()(index_t) const;
  };

  // Second push, updates D with J0 but assigns it to D0.
  template <>
  Inline void AmperePolesGR_init<Dim2>::operator()(index_t i) const {
    index_t j_min {N_GHOSTS};
    index_t j_max {m_nj + N_GHOSTS - 1};
    real_t  i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};

    real_t inv_polar_area_iPj {ONE / m_mblock.metric.polar_area({i_ + HALF, HALF})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, HALF})};

    // theta = 0
    DX1(i, j_min) += inv_polar_area_iPj * m_coeff * HX3(i, j_min);
    // theta = pi
    DX1(i, j_max + 1) -= inv_polar_area_iPj * m_coeff * HX3(i, j_max);
    // j = jmin + 1/2
    DX2(i, j_min) += inv_sqrt_detH_ijP * m_coeff * (HX3(i - 1, j_min) - HX3(i, j_min));
  }

} // namespace ntt

#endif
