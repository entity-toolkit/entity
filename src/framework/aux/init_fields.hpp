#ifndef GRPIC_INIT_FIELDS
#define GRPIC_INIT_FIELDS

#if (SIMTYPE == GRPIC_SIMTYPE)

#  include "global.h"
#  include "fields.h"
#  include "meshblock.h"

#  include "problem_generator.hpp"

#  include <stdexcept>

namespace ntt {

  /**
   * Computes D and B from A3, A1, A0
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class initFieldsFromVectorPotential {
    using index_t = typename RealFieldND<D, 6>::size_type;
    ProblemGenerator<D, SimulationType::GRPIC> m_pgen;
    Meshblock<D, SimulationType::GRPIC>        m_mblock;
    real_t                                     m_eps;
    index_t                                    j_min;

  public:
    initFieldsFromVectorPotential(const ProblemGenerator<D, SimulationType::GRPIC>& pgen,
                                  const Meshblock<D, SimulationType::GRPIC>&        mblock,
                                  real_t                                            eps)
      : m_pgen {pgen}, m_mblock {mblock}, m_eps {eps}, j_min {static_cast<index_t>(m_mblock.j_min())} {}

    Inline void operator()(const index_t, const index_t) const;
  };

  template <>
  Inline void initFieldsFromVectorPotential<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};

    coord_t<Dimension::TWO_D> x0m, x0p;

    real_t inv_sqrt_detH_ij {ONE / m_mblock.metric.sqrt_det_h({i_, j_})};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    real_t inv_sqrt_detH_iPj {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_iPjP {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_ + HALF})};
    real_t sqrt_detH_ij {m_mblock.metric.sqrt_det_h({i_, j_})};
    real_t sqrt_detH_ijP {m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    real_t alpha_ij {m_mblock.metric.alpha({i_, j_})};
    real_t alpha_ijP {m_mblock.metric.alpha({i_, j_ + HALF})};
    real_t alpha_iPj {m_mblock.metric.alpha({i_ + HALF, j_})};
    real_t beta_ij {m_mblock.metric.beta1u({i_, j_})};
    real_t beta_ijP {m_mblock.metric.beta1u({i_, j_ + HALF})};
    real_t h_11_inv_iPj {m_mblock.metric.h_11_inv({i_ + HALF, j_})};
    real_t h_13_inv_iPj {m_mblock.metric.h_13_inv({i_ + HALF, j_})};
    real_t h_22_inv_ijP {m_mblock.metric.h_22_inv({i_, j_ + HALF})};
    real_t h_33_inv_ij {m_mblock.metric.h_33_inv({i_, j_})};
    real_t h_13_inv_ij {m_mblock.metric.h_13_inv({i_, j_})};

    x0m[0] = i_;
    x0m[1] = j_ + HALF - HALF * m_eps;
    x0p[0] = i_;
    x0p[1] = j_ + HALF + HALF * m_eps;

    real_t E2d {(m_pgen.A0(m_mblock, x0p) - m_pgen.A0(m_mblock, x0m)) / m_eps};
    real_t B1u {(m_pgen.A3(m_mblock, x0p) - m_pgen.A3(m_mblock, x0m)) * inv_sqrt_detH_ijP / m_eps};
    real_t B3_aux {-(m_pgen.A1(m_mblock, x0p) - m_pgen.A1(m_mblock, x0m)) * inv_sqrt_detH_ijP / m_eps};

    x0m[0] = i_ + HALF - HALF * m_eps;
    x0m[1] = j_;
    x0p[0] = i_ + HALF + HALF * m_eps;
    x0p[1] = j_;

    real_t B2u;
    if (j == j_min) {
      B2u = ZERO;
    } else {
      B2u = -(m_pgen.A3(m_mblock, x0p) - m_pgen.A3(m_mblock, x0m)) * inv_sqrt_detH_iPj / m_eps;
    }
    real_t E1d {(m_pgen.A0(m_mblock, x0p) - m_pgen.A0(m_mblock, x0m)) / m_eps};

    x0m[0] = i_ + HALF;
    x0m[1] = j_ + HALF - HALF * m_eps;
    x0p[0] = i_ + HALF;
    x0p[1] = j_ + HALF + HALF * m_eps;

    real_t B3u {-(m_pgen.A1(m_mblock, x0p) - m_pgen.A1(m_mblock, x0m)) * inv_sqrt_detH_iPjP / m_eps};

    x0m[0] = i_ - HALF * m_eps;
    x0m[1] = j_;
    x0p[0] = i_ + HALF * m_eps;
    x0p[1] = j_;

    real_t B2_aux;
    if (j == j_min) {
      B2_aux = ZERO;
    } else {
      B2_aux = -(m_pgen.A3(m_mblock, x0p) - m_pgen.A3(m_mblock, x0m)) * inv_sqrt_detH_ij / m_eps;
    }

    // Compute covariant D
    real_t D1d {E1d / alpha_iPj};
    real_t D2d {E2d / alpha_ijP + sqrt_detH_ijP * beta_ijP * B3_aux / alpha_ijP};
    real_t D3d {-sqrt_detH_ij * beta_ij * B2_aux / alpha_ij};

    // Covariant D to contravariant D
    real_t D1u {h_11_inv_iPj * D1d + h_13_inv_iPj * D3d};
    real_t D2u {h_22_inv_ijP * D2d};
    real_t D3u;

    // h33_inv is singular at theta = 0.
    if (j == j_min) {
      D3u = ZERO;
    } else {
      D3u = h_33_inv_ij * D3d + h_13_inv_ij * D1d;
    }
    m_mblock.em(i, j, em::bx1) = B1u;
    m_mblock.em(i, j, em::bx2) = B2u;
    m_mblock.em(i, j, em::bx3) = B3u;
    m_mblock.em(i, j, em::ex1) = D1u;
    m_mblock.em(i, j, em::ex2) = D2u;
    m_mblock.em(i, j, em::ex3) = D3u;
  }

} // namespace ntt

#endif

#endif
