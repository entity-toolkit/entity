#ifndef FRAMEWORK_GENERATE_FIELDS_H
#define FRAMEWORK_GENERATE_FIELDS_H

#include "wrapper.h"

#include "sim_params.h"

#include "meshblock/meshblock.h"

#include "utils/archetypes.hpp"

namespace ntt {
  template <template <Dimension, SimulationEngine> class VectorPotential>
  class Generate2DGRFromVectorPotential_kernel {
    Meshblock<Dim2, GRPICEngine>       m_mblock;
    VectorPotential<Dim2, GRPICEngine> m_v_pot;
    real_t                             m_eps;
    index_t                            j_min;

  public:
    Generate2DGRFromVectorPotential_kernel(const SimulationParams&             params,
                                           const Meshblock<Dim2, GRPICEngine>& mblock,
                                           const real_t&                       eps)
      : m_mblock { mblock },
        m_v_pot { params, mblock },
        m_eps { eps },
        j_min { static_cast<index_t>(m_mblock.i2_min()) } {}

    Inline void operator()(index_t i, index_t j) const {
      real_t        i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
      real_t        j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

      coord_t<Dim2> x0m { ZERO }, x0p { ZERO };

      real_t        inv_sqrt_detH_ij { ONE / m_mblock.metric.sqrt_det_h({ i_, j_ }) };
      real_t        inv_sqrt_detH_ijP { ONE / m_mblock.metric.sqrt_det_h({ i_, j_ + HALF }) };
      real_t        inv_sqrt_detH_iPj { ONE / m_mblock.metric.sqrt_det_h({ i_ + HALF, j_ }) };
      real_t inv_sqrt_detH_iPjP { ONE / m_mblock.metric.sqrt_det_h({ i_ + HALF, j_ + HALF }) };
      real_t sqrt_detH_ij { m_mblock.metric.sqrt_det_h({ i_, j_ }) };
      real_t sqrt_detH_ijP { m_mblock.metric.sqrt_det_h({ i_, j_ + HALF }) };
      real_t alpha_ij { m_mblock.metric.alpha({ i_, j_ }) };
      real_t alpha_ijP { m_mblock.metric.alpha({ i_, j_ + HALF }) };
      real_t alpha_iPj { m_mblock.metric.alpha({ i_ + HALF, j_ }) };
      real_t beta_ij { m_mblock.metric.beta1({ i_, j_ }) };
      real_t beta_ijP { m_mblock.metric.beta1({ i_, j_ + HALF }) };
      real_t h11_iPj { m_mblock.metric.h11({ i_ + HALF, j_ }) };
      real_t h13_iPj { m_mblock.metric.h13({ i_ + HALF, j_ }) };
      real_t h22_ijP { m_mblock.metric.h22({ i_, j_ + HALF }) };
      real_t h33_ij { m_mblock.metric.h33({ i_, j_ }) };
      real_t h13_ij { m_mblock.metric.h13({ i_, j_ }) };

      x0m[0] = i_;
      x0m[1] = j_ + HALF - HALF * m_eps;
      x0p[0] = i_;
      x0p[1] = j_ + HALF + HALF * m_eps;

      real_t E2d { (m_v_pot.A_x0(x0p) - m_v_pot.A_x0(x0m)) / m_eps };
      real_t B1u { (m_v_pot.A_x3(x0p) - m_v_pot.A_x3(x0m)) * inv_sqrt_detH_ijP / m_eps };
      real_t B3_aux { -(m_v_pot.A_x1(x0p) - m_v_pot.A_x1(x0m)) * inv_sqrt_detH_ijP / m_eps };

      x0m[0] = i_ + HALF - HALF * m_eps;
      x0m[1] = j_;
      x0p[0] = i_ + HALF + HALF * m_eps;
      x0p[1] = j_;

      real_t B2u;
      if (j == j_min) {
        B2u = ZERO;
      } else {
        B2u = -(m_v_pot.A_x3(x0p) - m_v_pot.A_x3(x0m)) * inv_sqrt_detH_iPj / m_eps;
      }
      real_t E1d { (m_v_pot.A_x0(x0p) - m_v_pot.A_x0(x0m)) / m_eps };

      x0m[0] = i_ + HALF;
      x0m[1] = j_ + HALF - HALF * m_eps;
      x0p[0] = i_ + HALF;
      x0p[1] = j_ + HALF + HALF * m_eps;

      real_t B3u { -(m_v_pot.A_x1(x0p) - m_v_pot.A_x1(x0m)) * inv_sqrt_detH_iPjP / m_eps };

      x0m[0] = i_ - HALF * m_eps;
      x0m[1] = j_;
      x0p[0] = i_ + HALF * m_eps;
      x0p[1] = j_;

      real_t B2_aux;
      if (j == j_min) {
        B2_aux = ZERO;
      } else {
        B2_aux = -(m_v_pot.A_x3(x0p) - m_v_pot.A_x3(x0m)) * inv_sqrt_detH_ij / m_eps;
      }

      // Compute covariant D
      real_t D1d { E1d / alpha_iPj };
      real_t D2d { E2d / alpha_ijP + sqrt_detH_ijP * beta_ijP * B3_aux / alpha_ijP };
      real_t D3d { -sqrt_detH_ij * beta_ij * B2_aux / alpha_ij };

      // Covariant D to contravariant D
      real_t D1u { h11_iPj * D1d + h13_iPj * D3d };
      real_t D2u { h22_ijP * D2d };
      real_t D3u;

      // h33_inv is singular at theta = 0.
      if (j == j_min) {
        D3u = ZERO;
      } else {
        D3u = h33_ij * D3d + h13_ij * D1d;
      }
      m_mblock.em(i, j, em::bx1) = B1u;
      m_mblock.em(i, j, em::bx2) = B2u;
      m_mblock.em(i, j, em::bx3) = B3u;
      m_mblock.em(i, j, em::dx1) = D1u;
      m_mblock.em(i, j, em::dx2) = D2u;
      m_mblock.em(i, j, em::dx3) = D3u;
    }
  };
}    // namespace ntt

#endif    // FRAMEWORK_GRPIC_GENERATE_HPP