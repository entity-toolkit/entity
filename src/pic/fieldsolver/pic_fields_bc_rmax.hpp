#ifndef PIC_FIELDS_BC_RMAX_H
#define PIC_FIELDS_BC_RMAX_H

#include "global.h"
#include "pic.h"
#include "problem_generator.hpp"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {
  /**
   * @brief Algorithms for PIC field boundary conditions.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class FieldBC_rmax {
    Meshblock<D, SimulationType::PIC>        m_mblock;
    ProblemGenerator<D, SimulationType::PIC> m_pgen;
    real_t                                   m_rabsorb;
    real_t                                   m_rmax;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param pgen Problem generator.
     * @param rabsorb Absorbing radius.
     * @param rmax Maximum radius.
     */
    FieldBC_rmax(const Meshblock<D, SimulationType::PIC>&        mblock,
                 const ProblemGenerator<D, SimulationType::PIC>& pgen,
                 real_t                                          r_absorb,
                 real_t                                          r_max)
      : m_mblock {mblock}, m_pgen {pgen}, m_rabsorb {r_absorb}, m_rmax {r_max} {}
    /**
     * @brief 2D implementation of the algorithm.
     * @param i1 index.
     * @param i2 index.
     */
    Inline void operator()(index_t, index_t) const;
  };

  template <>
  Inline void FieldBC_rmax<Dimension::TWO_D>::operator()(index_t i,
                                                         index_t j) const {
    real_t i_ {static_cast<real_t>(i)};
    real_t j_ {static_cast<real_t>(j)};

    // i
    vec_t<Dimension::TWO_D> rth_;
    m_mblock.metric.x_Code2Sph({i_, j_}, rth_);
    real_t delta_r1 {(rth_[0] - m_rabsorb) / (m_rmax - m_rabsorb)};
    real_t sigma_r1 {HEAVISIDE(delta_r1) * delta_r1 * delta_r1 * delta_r1};
    // i + 1/2
    m_mblock.metric.x_Code2Sph({i_ + HALF, j_}, rth_);
    real_t delta_r2 {(rth_[0] - m_rabsorb) / (m_rmax - m_rabsorb)};
    real_t sigma_r2 {HEAVISIDE(delta_r2) * delta_r2 * delta_r2 * delta_r2};

    m_mblock.em(i, j, em::ex1) = (ONE - sigma_r2) * m_mblock.em(i, j, em::ex1);
    m_mblock.em(i, j, em::bx2) = (ONE - sigma_r2) * m_mblock.em(i, j, em::bx2);
    m_mblock.em(i, j, em::bx3) = (ONE - sigma_r2) * m_mblock.em(i, j, em::bx3);

    real_t br_target_hat {m_pgen.userTargetField_br_hat(m_mblock, {i_, j_ + HALF})};
    real_t bx1_source_cntr {m_mblock.em(i, j, em::bx1)};
    vec_t<Dimension::THREE_D> br_source_hat;
    m_mblock.metric.v_Cntrv2Hat({i_, j_ + HALF}, {bx1_source_cntr, ZERO, ZERO}, br_source_hat);
    real_t br_interm_hat {(ONE - sigma_r1) * br_source_hat[0] + sigma_r1 * br_target_hat};
    vec_t<Dimension::THREE_D> br_interm_cntr;
    m_mblock.metric.v_Hat2Cntrv({i_, j_ + HALF}, {br_interm_hat, ZERO, ZERO}, br_interm_cntr);
    m_mblock.em(i, j, em::bx1) = br_interm_cntr[0];
    m_mblock.em(i, j, em::ex2) = (ONE - sigma_r1) * m_mblock.em(i, j, em::ex2);
    m_mblock.em(i, j, em::ex3) = (ONE - sigma_r1) * m_mblock.em(i, j, em::ex3);
  }
} // namespace ntt

#endif
