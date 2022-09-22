#ifndef GRPIC_FIELDS_BC_H
#define GRPIC_FIELDS_BC_H

#include "global.h"
#include "grpic.h"
#include "problem_generator.hpp"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {
  /**
   * Algorithms for GRPIC field boundary conditions.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class GRFieldBC_rmax {

    Meshblock<D, SimulationType::GRPIC>        m_mblock;
    ProblemGenerator<D, SimulationType::GRPIC> m_pgen;
    real_t                                     m_rabsorb;
    real_t                                     m_rmax;
    real_t                                     m_absorbcoeff;
    real_t                                     m_absorbnorm;

  public:
    GRFieldBC_rmax(const Meshblock<D, SimulationType::GRPIC>&        mblock,
                   const ProblemGenerator<D, SimulationType::GRPIC>& pgen,
                   real_t                                            r_absorb,
                   real_t                                            r_max,
                   real_t                                            absorb_coeff,
                   real_t                                            absorb_norm)
      : m_mblock {mblock},
        m_pgen {pgen},
        m_rabsorb {r_absorb},
        m_rmax {r_max},
        m_absorbcoeff {absorb_coeff},
        m_absorbnorm {absorb_norm} {}
    Inline void operator()(index_t, index_t) const;
  };

  template <>
  Inline void GRFieldBC_rmax<Dim2>::operator()(index_t i, index_t j) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};

    // i
    vec_t<Dim2> rth_;
    m_mblock.metric.x_Code2Sph({i_, j_}, rth_);
    if (rth_[0] > m_rabsorb) {
      real_t delta_r1 {(rth_[0] - m_rabsorb) / (m_rmax - m_rabsorb)};
      real_t sigma_r1 {
        m_absorbnorm
        * (ONE - math::exp(m_absorbcoeff * HEAVISIDE(delta_r1) * CUBE(delta_r1)))};
      real_t br_target {m_pgen.userTargetField_br_cntrv(m_mblock, {i_, j_ + HALF})};
      m_mblock.em0(i, j, em::bx1)
        = (ONE - sigma_r1) * m_mblock.em0(i, j, em::bx1) + sigma_r1 * br_target;
      m_mblock.em(i, j, em::bx1)
        = (ONE - sigma_r1) * m_mblock.em(i, j, em::bx1) + sigma_r1 * br_target;
    }
    // i + 1/2
    m_mblock.metric.x_Code2Sph({i_ + HALF, j_}, rth_);
    if (rth_[0] > m_rabsorb) {
      real_t delta_r2 {(rth_[0] - m_rabsorb) / (m_rmax - m_rabsorb)};
      real_t sigma_r2 {
        m_absorbnorm
        * (ONE - math::exp(m_absorbcoeff * HEAVISIDE(delta_r2) * CUBE(delta_r2)))};
      real_t bth_target {m_pgen.userTargetField_bth_cntrv(m_mblock, {i_ + HALF, j_})};
      m_mblock.em0(i, j, em::bx2)
        = (ONE - sigma_r2) * m_mblock.em0(i, j, em::bx2) + sigma_r2 * bth_target;
      m_mblock.em(i, j, em::bx2)
        = (ONE - sigma_r2) * m_mblock.em(i, j, em::bx2) + sigma_r2 * bth_target;
      m_mblock.em0(i, j, em::bx3) = (ONE - sigma_r2) * m_mblock.em0(i, j, em::bx3);
      m_mblock.em(i, j, em::bx3)  = (ONE - sigma_r2) * m_mblock.em(i, j, em::bx3);
    }
  }
} // namespace ntt

#endif // GRPIC_HPP
