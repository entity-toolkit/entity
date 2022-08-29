#include "global.h"
#include "pic.h"

#include "particle_macros.h"

#include <plog/Log.h>

#include <stdexcept>
#include <iostream>

namespace ntt {
  /**
   * @brief 1d periodic particle bc.
   *
   */
  template <>
  void PIC<Dimension::ONE_D>::particleBoundaryConditions(const real_t&) {

#if (METRIC == MINKOWSKI_METRIC)
    if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      for (auto& species : m_mblock.particles) {
        auto ni {m_mblock.Ni1()};
        Kokkos::parallel_for(
          "prtl_bc", species.loopParticles(), Lambda(index_t p) {
            if (species.i1(p) < 0) {
              species.i1(p) += ni;
            } else if (species.i1(p) >= ni) {
              species.i1(p) -= ni;
            }
          });
      }
    } else {
      for (auto& species : m_mblock.particles) {
        auto ni {m_mblock.Ni1()};
        Kokkos::parallel_for(
          "prtl_bc", species.loopParticles(), Lambda(index_t p) {
            species.is_dead(p) = ((species.i1(p) < 0) || (species.i1(p) >= ni));
          });
      }
    }
#else
    NTTError("only minkowski possible in 1d");
#endif
  }

  /**
   * @brief 2d periodic particle bc.
   *
   */
  template <>
  void PIC<Dimension::TWO_D>::particleBoundaryConditions(const real_t&) {

#if (METRIC == MINKOWSKI_METRIC)
    if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      for (auto& species : m_mblock.particles) {
        auto ni {m_mblock.Ni1()};
        auto nj {m_mblock.Ni2()};
        Kokkos::parallel_for(
          "prtl_bc", species.loopParticles(), Lambda(index_t p) {
            if (species.i1(p) < 0) {
              species.i1(p) += ni;
            } else if (species.i1(p) >= ni) {
              species.i1(p) -= ni;
            }
            if (species.i2(p) < 0) {
              species.i2(p) += nj;
            } else if (species.i2(p) >= nj) {
              species.i2(p) -= nj;
            }
          });
      }
    } else {
      for (auto& species : m_mblock.particles) {
        auto ni {m_mblock.Ni1()};
        auto nj {m_mblock.Ni2()};
        Kokkos::parallel_for(
          "prtl_bc", species.loopParticles(), Lambda(index_t p) {
            species.is_dead(p) = ((species.i1(p) < 0) || (species.i1(p) >= ni)
                                  || (species.i2(p) < 0) || (species.i2(p) >= nj));
          });
      }
    }
#else
    for (auto& species : m_mblock.particles) {
      auto ni {m_mblock.Ni1()};
      auto nj {m_mblock.Ni2()};
      Kokkos::parallel_for(
        "prtl_bc", species.loopParticles(), Lambda(index_t p) {
          // radial boundary conditions
          species.is_dead(p) = ((species.i1(p) < -1) || (species.i1(p) >= ni + 1));
          if (species.i2(p) < 0) {
            // reflect particle coordinate
            species.i2(p)  = 0;
            species.dx2(p) = 1.0f - species.dx2(p);
            // reverse u^theta
            coord_t<Dim3> x_p {PRTL_X1(species, p), PRTL_X2(species, p), species.phi(p)};
            vec_t<Dim3>   u_hat, u_cart;
            m_mblock.metric.v_Cart2Hat(
              x_p, {species.ux1(p), species.ux2(p), species.ux3(p)}, u_hat);
            m_mblock.metric.v_Hat2Cart(x_p, {u_hat[0], -u_hat[1], u_hat[2]}, u_cart);
            species.ux1(p) = u_cart[0];
            species.ux2(p) = u_cart[1];
            species.ux3(p) = u_cart[2];
          }
          if (species.i2(p) >= nj) {
            // reflect particle coordinate
            species.i2(p)  = nj - 1;
            species.dx2(p) = 1.0f - species.dx2(p);
            // reverse u^theta
            coord_t<Dim3> x_p {PRTL_X1(species, p), PRTL_X2(species, p), species.phi(p)};
            vec_t<Dim3>   u_hat, u_cart;
            m_mblock.metric.v_Cart2Hat(
              x_p, {species.ux1(p), species.ux2(p), species.ux3(p)}, u_hat);
            m_mblock.metric.v_Hat2Cart(x_p, {u_hat[0], -u_hat[1], u_hat[2]}, u_cart);
            species.ux1(p) = u_cart[0];
            species.ux2(p) = u_cart[1];
            species.ux3(p) = u_cart[2];
          }
        });
    }
#endif
  }

  /**
   * @brief 3d periodic particle bc.
   *
   */
  template <>
  void PIC<Dimension::THREE_D>::particleBoundaryConditions(const real_t&) {
    NTTError("not implemented");
  }

} // namespace ntt
