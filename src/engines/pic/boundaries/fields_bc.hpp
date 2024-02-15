#ifndef PIC_FIELDS_BC_H
#define PIC_FIELDS_BC_H

#include "wrapper.h"

#include "field_macros.h"
#include "pic.h"
#include "sim_params.h"

#include "utils/archetypes.hpp"
#include PGEN_HEADER

namespace ntt {
  /**
   * @brief Algorithms for PIC field boundary conditions.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class AbsorbFields_kernel {
    SimulationParams               m_params;
    Meshblock<D, PICEngine>        m_mblock;
    real_t                         m_rabsorb;
    real_t                         m_rmax;
    const std::size_t              m_i2_min;
    PgenTargetFields<D, PICEngine> m_target_fields;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param pgen Problem generator.
     * @param rabsorb Absorbing radius.
     * @param rmax Maximum radius.
     */
    AbsorbFields_kernel(const SimulationParams&        params,
                        const Meshblock<D, PICEngine>& mblock,
                        real_t                         r_absorb,
                        real_t                         r_max) :
      m_params { params },
      m_mblock { mblock },
      m_rabsorb { r_absorb },
      m_rmax { r_max },
      m_i2_min { m_mblock.i2_min() },
      m_target_fields { m_params, m_mblock } {}

    ~AbsorbFields_kernel() {}

    /**
     * @brief 2D implementation of the algorithm.
     * @param i1 index.
     * @param i2 index.
     */
    Inline void operator()(index_t, index_t) const;
  };

  template <>
  Inline void AbsorbFields_kernel<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

    // i
    real_t delta_r1 { (m_mblock.metric.x1_Code2Sph(i_) - m_rabsorb) /
                      (m_rmax - m_rabsorb) };
    real_t sigma_r1 { HEAVISIDE(delta_r1) * CUBE(delta_r1) };
    // i + 1/2
    real_t delta_r2 { (m_mblock.metric.x1_Code2Sph(i_ + HALF) - m_rabsorb) /
                      (m_rmax - m_rabsorb) };
    real_t sigma_r2 { HEAVISIDE(delta_r2) * CUBE(delta_r2) };

    // components at the axes treated separately
    if (j == m_i2_min) {
      EX1(i, j) = ZERO;
      EX3(i, j) = ZERO;
      BX2(i, j) = ZERO;
    } else {
      // Ex1 : i + 1/2, j
      EX1(i, j) = ((ONE - sigma_r2) *
                     (math::sqrt(m_mblock.metric.h_11({ i_ + HALF, j_ })) *
                      EX1(i, j)) +
                   sigma_r2 * m_target_fields(em::ex1, { i_ + HALF, j_ })) /
                  math::sqrt(m_mblock.metric.h_11({ i_ + HALF, j_ }));
      // Ex3 : i, j
      EX3(i, j) = ((ONE - sigma_r1) *
                     (math::sqrt(m_mblock.metric.h_33({ i_, j_ })) * EX3(i, j)) +
                   sigma_r1 * m_target_fields(em::ex3, { i_, j_ })) /
                  math::sqrt(m_mblock.metric.h_33({ i_, j_ }));
      // Bx2 : i + 1/2, j
      BX2(i, j) = ((ONE - sigma_r2) *
                     (math::sqrt(m_mblock.metric.h_22({ i_ + HALF, j_ })) *
                      BX2(i, j)) +
                   sigma_r2 * m_target_fields(em::bx2, { i_ + HALF, j_ })) /
                  math::sqrt(m_mblock.metric.h_22({ i_ + HALF, j_ }));
    }

    // Ex2 : i, j + 1/2
    EX2(i, j) = ((ONE - sigma_r1) *
                   (math::sqrt(m_mblock.metric.h_22({ i_, j_ + HALF })) * EX2(i, j)) +
                 sigma_r1 * m_target_fields(em::ex2, { i_, j_ + HALF })) /
                math::sqrt(m_mblock.metric.h_22({ i_, j_ + HALF }));

    // Bx1 : i, j + 1/2
    BX1(i, j) = ((ONE - sigma_r1) *
                   (math::sqrt(m_mblock.metric.h_11({ i_, j_ + HALF })) * BX1(i, j)) +
                 sigma_r1 * m_target_fields(em::bx1, { i_, j_ + HALF })) /
                math::sqrt(m_mblock.metric.h_11({ i_, j_ + HALF }));

    // Bx3 : i + 1/2, j + 1/2
    BX3(i, j) = ((ONE - sigma_r2) *
                   (math::sqrt(m_mblock.metric.h_33({ i_ + HALF, j_ + HALF })) *
                    BX3(i, j)) +
                 sigma_r2 * m_target_fields(em::bx3, { i_ + HALF, j_ + HALF })) /
                math::sqrt(m_mblock.metric.h_33({ i_ + HALF, j_ + HALF }));
  }
} // namespace ntt

#endif
