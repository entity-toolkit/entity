#ifndef GRPIC_FIELDS_BC_H
#define GRPIC_FIELDS_BC_H

#include "wrapper.h"

#include "field_macros.h"
#include "grpic.h"
#include "sim_params.h"

#include "problem_generator.hpp"

namespace ntt {
  /**
   * Algorithms for GRPIC field boundary conditions.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class AbsorbFieldsRmax_kernel {
    SimulationParams                 m_params;
    Meshblock<D, GRPICEngine>        m_mblock;
    ProblemGenerator<D, GRPICEngine> m_pgen;
    const real_t                     m_rabsorb;
    const real_t                     m_rmax;
    const real_t                     m_absorbcoeff;
    const real_t                     m_absorbnorm;

  public:
    AbsorbFieldsRmax_kernel(const SimulationParams&                 params,
                            const Meshblock<D, GRPICEngine>&        mblock,
                            const ProblemGenerator<D, GRPICEngine>& pgen,
                            real_t                                  r_absorb,
                            real_t                                  r_max)
      : m_params { params },
        m_mblock { mblock },
        m_pgen { pgen },
        m_rabsorb { r_absorb },
        m_rmax { r_max },
        m_absorbcoeff { m_params.metricParameters()[3] },
        m_absorbnorm { ONE / (ONE - math::exp(m_absorbcoeff)) } {}
    Inline void operator()(index_t, index_t) const;
  };

  template <>
  Inline void AbsorbFieldsRmax_kernel<Dim2>::operator()(index_t i, index_t j) const {
    real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    real_t      j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

    // i
    vec_t<Dim2> rth_;
    m_mblock.metric.x_Code2Sph({ i_, j_ }, rth_);
    if (rth_[0] > m_rabsorb) {
      real_t delta_r1 { (rth_[0] - m_rabsorb) / (m_rmax - m_rabsorb) };
      real_t sigma_r1 {
        m_absorbnorm * (ONE - math::exp(m_absorbcoeff * HEAVISIDE(delta_r1) * CUBE(delta_r1)))
      };
      // real_t br_target { m_pgen.userTargetField_br_cntrv(m_mblock, { i_, j_ + HALF }) };
      // B0X1(i, j) = (ONE - sigma_r1) * B0X1(i, j) + sigma_r1 * br_target;
      // BX1(i, j)  = (ONE - sigma_r1) * BX1(i, j) + sigma_r1 * br_target;
    }
    // i + 1/2
    m_mblock.metric.x_Code2Sph({ i_ + HALF, j_ }, rth_);
    if (rth_[0] > m_rabsorb) {
      real_t delta_r2 { (rth_[0] - m_rabsorb) / (m_rmax - m_rabsorb) };
      real_t sigma_r2 {
        m_absorbnorm * (ONE - math::exp(m_absorbcoeff * HEAVISIDE(delta_r2) * CUBE(delta_r2)))
      };
      // real_t bth_target { m_pgen.userTargetField_bth_cntrv(m_mblock, { i_ + HALF, j_ }) };
      // B0X2(i, j) = (ONE - sigma_r2) * B0X2(i, j) + sigma_r2 * bth_target;
      // BX2(i, j)  = (ONE - sigma_r2) * BX2(i, j) + sigma_r2 * bth_target;
      // B0X3(i, j) = (ONE - sigma_r2) * B0X3(i, j);
      // BX3(i, j)  = (ONE - sigma_r2) * BX3(i, j);
    }
  }
}    // namespace ntt

#endif    // GRPIC_HPP
