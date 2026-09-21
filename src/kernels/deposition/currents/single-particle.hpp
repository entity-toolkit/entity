/**
 * @file kernels/deposition/currents/single-particle.hpp
 * @brief Per-particle current deposition math, shared between the flat and tiled kernels.
 *
 * @implements
 *   - kernel::DepositOneParticle<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_DEPOSITION_CURRENTS_SINGLE_PARTICLE_HPP
#define KERNELS_DEPOSITION_CURRENTS_SINGLE_PARTICLE_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "kernels/particle_shapes.hpp"

#include <Kokkos_Core.hpp>

#define i_di_to_Xi(I, DI) (static_cast<real_t>((I)) + static_cast<real_t>((DI)))

namespace kernel {
  using namespace ntt;

  /**
   * @brief Per-particle deposit body, shared between the flat and tiled
   *        kernels.
   *
   * The caller supplies a `deposit_at(idx..., comp, val)` callback that
   * applies the contribution `val` to the J component `comp` at the
   * **global** J cell index `idx...` (already includes the `N_GHOSTS`
   * offset). The flat kernel's callback simply does
   * `J_acc(idx..., comp) += val` on its scatter-view accessor; the tiled
   * kernel's callback translates `idx...` into per-tile scratch
   * coordinates and uses `Kokkos::atomic_add` on SLM. Either way, this
   * function is identical numerically and contains the only deposit math
   * in the codebase.
   *
   * Dead particles return early. The callback is invoked once per cell
   * write, with the dimension-appropriate signature:
   *   - 1D: `deposit_at(int g_i1, int comp, real_t val)`
   *   - 2D: `deposit_at(int g_i1, int g_i2, int comp, real_t val)`
   *   - 3D: `deposit_at(int g_i1, int g_i2, int g_i3, int comp, real_t val)`
   */
  template <SimEngine::type S, MetricClass M, unsigned short O, typename DepositFn>
  Inline void DepositOneParticle(prtlidx_t             p,
                                 const ParticleArrays& prtls,
                                 const M&              metric,
                                 real_t                charge,
                                 real_t                inv_dt,
                                 DepositFn             deposit_at) {
    static_assert(O <= 11u, "Shape function order O must be <= 11");
    constexpr auto D = M::Dim;

    if (prtls.tag(p) == ParticleTag::dead) {
      return;
    }

    // recover particle velocity to deposit in unsimulated direction
    [[maybe_unused]]
    vec_t<Dim::_3D> vp { ZERO };
    // `vp` only feeds the unsimulated-direction current in the 1D
    // (jx2, jx3) and 2D (jx3) branches. In 3D every J component comes
    // from the Esirkepov/zigzag charge motion and `vp` is never read,
    // so the metric transform + 1/sqrt + NaN/Inf guard below is pure
    // dead work there — skip it (also frees xp/inv_energy registers).
    if constexpr (D != Dim::_3D) {
      coord_t<M::PrtlDim> xp { ZERO };
      if constexpr (D == Dim::_1D) {
        xp[0] = i_di_to_Xi(prtls.i1(p), prtls.dx1(p));
      } else if constexpr (D == Dim::_2D) {
        if constexpr (M::PrtlDim == Dim::_3D) {
          xp[0] = i_di_to_Xi(prtls.i1(p), prtls.dx1(p));
          xp[1] = i_di_to_Xi(prtls.i2(p), prtls.dx2(p));
          xp[2] = prtls.phi(p);
        } else {
          xp[0] = i_di_to_Xi(prtls.i1(p), prtls.dx1(p));
          xp[1] = i_di_to_Xi(prtls.i2(p), prtls.dx2(p));
        }
      } else {
        xp[0] = i_di_to_Xi(prtls.i1(p), prtls.dx1(p));
        xp[1] = i_di_to_Xi(prtls.i2(p), prtls.dx2(p));
        xp[2] = i_di_to_Xi(prtls.i3(p), prtls.dx3(p));
      }
      auto inv_energy { ZERO };
      if constexpr (S == SimEngine::SRPIC) {
        metric.template transform_xyz<Idx::XYZ, Idx::U>(
          xp,
          { prtls.ux1(p), prtls.ux2(p), prtls.ux3(p) },
          vp);
        inv_energy = ONE / U2GAMMA(prtls.ux1(p), prtls.ux2(p), prtls.ux3(p));
      } else {
        coord_t<Dim::_2D> xp_ { ZERO };
        xp_[0] = xp[0];
        real_t     theta_Cd { xp[1] };
        const auto theta_Ph { metric.template convert<2, Crd::Cd, Crd::Ph>(
          theta_Cd) };
        const auto small_angle { static_cast<real_t>(constant::SMALL_ANGLE_GR) };
        const auto large_angle { static_cast<real_t>(
          constant::PI - constant::SMALL_ANGLE_GR) };
        if (theta_Ph < small_angle) {
          theta_Cd = metric.template convert<2, Crd::Ph, Crd::Cd>(small_angle);
        } else if (theta_Ph >= large_angle) {
          theta_Cd = metric.template convert<2, Crd::Ph, Crd::Cd>(large_angle);
        }
        xp_[1] = theta_Cd;
        metric.template transform<Idx::D, Idx::U>(
          xp_,
          { prtls.ux1(p), prtls.ux2(p), prtls.ux3(p) },
          vp);
        inv_energy = metric.alpha(xp_) /
                     math::sqrt(ONE + prtls.ux1(p) * vp[0] +
                                prtls.ux2(p) * vp[1] + prtls.ux3(p) * vp[2]);
      }
      if (Kokkos::isnan(vp[2]) || Kokkos::isinf(vp[2])) {
        vp[2] = ZERO;
      }
      vp[0] *= inv_energy;
      vp[1] *= inv_energy;
      vp[2] *= inv_energy;
    }

    const real_t coeff { prtls.weight(p) * charge };

    if constexpr (O == 0u) {
      /*
        Zig-zag deposit
      */
      const auto dxp_r_1 { static_cast<prtldx_t>(prtls.i1(p) == prtls.i1_prev(p)) *
                           (prtls.dx1(p) + prtls.dx1_prev(p)) *
                           static_cast<prtldx_t>(INV_2) };

      const real_t Wx1_1 { INV_2 *
                           (dxp_r_1 + prtls.dx1_prev(p) +
                            static_cast<real_t>(prtls.i1(p) > prtls.i1_prev(p))) };
      const real_t Wx1_2 { INV_2 *
                           (prtls.dx1(p) + dxp_r_1 +
                            static_cast<real_t>(
                              static_cast<int>(prtls.i1(p) > prtls.i1_prev(p)) +
                              prtls.i1_prev(p) - prtls.i1(p))) };
      const real_t Fx1_1 { (static_cast<real_t>(prtls.i1(p) > prtls.i1_prev(p)) +
                            dxp_r_1 - prtls.dx1_prev(p)) *
                           coeff * inv_dt };
      const real_t Fx1_2 { (static_cast<real_t>(
                              prtls.i1(p) - prtls.i1_prev(p) -
                              static_cast<int>(prtls.i1(p) > prtls.i1_prev(p))) +
                            prtls.dx1(p) - dxp_r_1) *
                           coeff * inv_dt };

      if constexpr (D == Dim::_1D) {
        const real_t Fx2_1 { HALF * vp[1] * coeff };
        const real_t Fx2_2 { HALF * vp[1] * coeff };

        const real_t Fx3_1 { HALF * vp[2] * coeff };
        const real_t Fx3_2 { HALF * vp[2] * coeff };

        deposit_at(prtls.i1_prev(p) + N_GHOSTS, cur::jx1, Fx1_1);
        deposit_at(prtls.i1(p) + N_GHOSTS, cur::jx1, Fx1_2);

        deposit_at(prtls.i1_prev(p) + N_GHOSTS, cur::jx2, Fx2_1 * (ONE - Wx1_1));
        deposit_at(prtls.i1_prev(p) + N_GHOSTS + 1, cur::jx2, Fx2_1 * Wx1_1);
        deposit_at(prtls.i1(p) + N_GHOSTS, cur::jx2, Fx2_2 * (ONE - Wx1_2));
        deposit_at(prtls.i1(p) + N_GHOSTS + 1, cur::jx2, Fx2_2 * Wx1_2);

        deposit_at(prtls.i1_prev(p) + N_GHOSTS, cur::jx3, Fx3_1 * (ONE - Wx1_1));
        deposit_at(prtls.i1_prev(p) + N_GHOSTS + 1, cur::jx3, Fx3_1 * Wx1_1);
        deposit_at(prtls.i1(p) + N_GHOSTS, cur::jx3, Fx3_2 * (ONE - Wx1_2));
        deposit_at(prtls.i1(p) + N_GHOSTS + 1, cur::jx3, Fx3_2 * Wx1_2);
      } else if constexpr (D == Dim::_2D || D == Dim::_3D) {
        const auto dxp_r_2 { static_cast<prtldx_t>(prtls.i2(p) == prtls.i2_prev(p)) *
                             (prtls.dx2(p) + prtls.dx2_prev(p)) *
                             static_cast<prtldx_t>(INV_2) };

        const real_t Wx2_1 { INV_2 * (dxp_r_2 + prtls.dx2_prev(p) +
                                      static_cast<real_t>(prtls.i2(p) >
                                                          prtls.i2_prev(p))) };
        const real_t Wx2_2 { INV_2 *
                             (prtls.dx2(p) + dxp_r_2 +
                              static_cast<real_t>(
                                static_cast<int>(prtls.i2(p) > prtls.i2_prev(p)) +
                                prtls.i2_prev(p) - prtls.i2(p))) };
        const real_t Fx2_1 { (static_cast<real_t>(prtls.i2(p) > prtls.i2_prev(p)) +
                              dxp_r_2 - prtls.dx2_prev(p)) *
                             coeff * inv_dt };
        const real_t Fx2_2 {
          (static_cast<real_t>(prtls.i2(p) - prtls.i2_prev(p) -
                               static_cast<int>(prtls.i2(p) > prtls.i2_prev(p))) +
           prtls.dx2(p) - dxp_r_2) *
          coeff * inv_dt
        };

        if constexpr (D == Dim::_2D) {
          const real_t Fx3_1 { HALF * vp[2] * coeff };
          const real_t Fx3_2 { HALF * vp[2] * coeff };

          deposit_at(prtls.i1_prev(p) + N_GHOSTS,
                     prtls.i2_prev(p) + N_GHOSTS,
                     cur::jx1,
                     Fx1_1 * (ONE - Wx2_1));
          deposit_at(prtls.i1_prev(p) + N_GHOSTS,
                     prtls.i2_prev(p) + N_GHOSTS + 1,
                     cur::jx1,
                     Fx1_1 * Wx2_1);
          deposit_at(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS,
                     cur::jx1,
                     Fx1_2 * (ONE - Wx2_2));
          deposit_at(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS + 1,
                     cur::jx1,
                     Fx1_2 * Wx2_2);

          deposit_at(prtls.i1_prev(p) + N_GHOSTS,
                     prtls.i2_prev(p) + N_GHOSTS,
                     cur::jx2,
                     Fx2_1 * (ONE - Wx1_1));
          deposit_at(prtls.i1_prev(p) + N_GHOSTS + 1,
                     prtls.i2_prev(p) + N_GHOSTS,
                     cur::jx2,
                     Fx2_1 * Wx1_1);
          deposit_at(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS,
                     cur::jx2,
                     Fx2_2 * (ONE - Wx1_2));
          deposit_at(prtls.i1(p) + N_GHOSTS + 1,
                     prtls.i2(p) + N_GHOSTS,
                     cur::jx2,
                     Fx2_2 * Wx1_2);

          deposit_at(prtls.i1_prev(p) + N_GHOSTS,
                     prtls.i2_prev(p) + N_GHOSTS,
                     cur::jx3,
                     Fx3_1 * (ONE - Wx1_1) * (ONE - Wx2_1));
          deposit_at(prtls.i1_prev(p) + N_GHOSTS + 1,
                     prtls.i2_prev(p) + N_GHOSTS,
                     cur::jx3,
                     Fx3_1 * Wx1_1 * (ONE - Wx2_1));
          deposit_at(prtls.i1_prev(p) + N_GHOSTS,
                     prtls.i2_prev(p) + N_GHOSTS + 1,
                     cur::jx3,
                     Fx3_1 * (ONE - Wx1_1) * Wx2_1);
          deposit_at(prtls.i1_prev(p) + N_GHOSTS + 1,
                     prtls.i2_prev(p) + N_GHOSTS + 1,
                     cur::jx3,
                     Fx3_1 * Wx1_1 * Wx2_1);

          deposit_at(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS,
                     cur::jx3,
                     Fx3_2 * (ONE - Wx1_2) * (ONE - Wx2_2));
          deposit_at(prtls.i1(p) + N_GHOSTS + 1,
                     prtls.i2(p) + N_GHOSTS,
                     cur::jx3,
                     Fx3_2 * Wx1_2 * (ONE - Wx2_2));
          deposit_at(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS + 1,
                     cur::jx3,
                     Fx3_2 * (ONE - Wx1_2) * Wx2_2);
          deposit_at(prtls.i1(p) + N_GHOSTS + 1,
                     prtls.i2(p) + N_GHOSTS + 1,
                     cur::jx3,
                     Fx3_2 * Wx1_2 * Wx2_2);
        } else {
          const auto dxp_r_3 {
            static_cast<prtldx_t>(prtls.i3(p) == prtls.i3_prev(p)) *
            (prtls.dx3(p) + prtls.dx3_prev(p)) * static_cast<prtldx_t>(INV_2)
          };
          const real_t Wx3_1 { INV_2 * (dxp_r_3 + prtls.dx3_prev(p) +
                                        static_cast<real_t>(
                                          prtls.i3(p) > prtls.i3_prev(p))) };
          const real_t Wx3_2 {
            INV_2 *
            (prtls.dx3(p) + dxp_r_3 +
             static_cast<real_t>(static_cast<int>(prtls.i3(p) > prtls.i3_prev(p)) +
                                 prtls.i3_prev(p) - prtls.i3(p)))
          };
          const real_t Fx3_1 { (static_cast<real_t>(prtls.i3(p) > prtls.i3_prev(p)) +
                                dxp_r_3 - prtls.dx3_prev(p)) *
                               coeff * inv_dt };
          const real_t Fx3_2 {
            (static_cast<real_t>(prtls.i3(p) - prtls.i3_prev(p) -
                                 static_cast<int>(prtls.i3(p) > prtls.i3_prev(p))) +
             prtls.dx3(p) - dxp_r_3) *
            coeff * inv_dt
          };

          deposit_at(prtls.i1_prev(p) + N_GHOSTS,
                     prtls.i2_prev(p) + N_GHOSTS,
                     prtls.i3_prev(p) + N_GHOSTS,
                     cur::jx1,
                     Fx1_1 * (ONE - Wx2_1) * (ONE - Wx3_1));
          deposit_at(prtls.i1_prev(p) + N_GHOSTS,
                     prtls.i2_prev(p) + N_GHOSTS + 1,
                     prtls.i3_prev(p) + N_GHOSTS,
                     cur::jx1,
                     Fx1_1 * Wx2_1 * (ONE - Wx3_1));
          deposit_at(prtls.i1_prev(p) + N_GHOSTS,
                     prtls.i2_prev(p) + N_GHOSTS,
                     prtls.i3_prev(p) + N_GHOSTS + 1,
                     cur::jx1,
                     Fx1_1 * (ONE - Wx2_1) * Wx3_1);
          deposit_at(prtls.i1_prev(p) + N_GHOSTS,
                     prtls.i2_prev(p) + N_GHOSTS + 1,
                     prtls.i3_prev(p) + N_GHOSTS + 1,
                     cur::jx1,
                     Fx1_1 * Wx2_1 * Wx3_1);

          deposit_at(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS,
                     prtls.i3(p) + N_GHOSTS,
                     cur::jx1,
                     Fx1_2 * (ONE - Wx2_2) * (ONE - Wx3_2));
          deposit_at(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS + 1,
                     prtls.i3(p) + N_GHOSTS,
                     cur::jx1,
                     Fx1_2 * Wx2_2 * (ONE - Wx3_2));
          deposit_at(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS,
                     prtls.i3(p) + N_GHOSTS + 1,
                     cur::jx1,
                     Fx1_2 * (ONE - Wx2_2) * Wx3_2);
          deposit_at(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS + 1,
                     prtls.i3(p) + N_GHOSTS + 1,
                     cur::jx1,
                     Fx1_2 * Wx2_2 * Wx3_2);

          deposit_at(prtls.i1_prev(p) + N_GHOSTS,
                     prtls.i2_prev(p) + N_GHOSTS,
                     prtls.i3_prev(p) + N_GHOSTS,
                     cur::jx2,
                     Fx2_1 * (ONE - Wx1_1) * (ONE - Wx3_1));
          deposit_at(prtls.i1_prev(p) + N_GHOSTS + 1,
                     prtls.i2_prev(p) + N_GHOSTS,
                     prtls.i3_prev(p) + N_GHOSTS,
                     cur::jx2,
                     Fx2_1 * Wx1_1 * (ONE - Wx3_1));
          deposit_at(prtls.i1_prev(p) + N_GHOSTS,
                     prtls.i2_prev(p) + N_GHOSTS,
                     prtls.i3_prev(p) + N_GHOSTS + 1,
                     cur::jx2,
                     Fx2_1 * (ONE - Wx1_1) * Wx3_1);
          deposit_at(prtls.i1_prev(p) + N_GHOSTS + 1,
                     prtls.i2_prev(p) + N_GHOSTS,
                     prtls.i3_prev(p) + N_GHOSTS + 1,
                     cur::jx2,
                     Fx2_1 * Wx1_1 * Wx3_1);

          deposit_at(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS,
                     prtls.i3(p) + N_GHOSTS,
                     cur::jx2,
                     Fx2_2 * (ONE - Wx1_2) * (ONE - Wx3_2));
          deposit_at(prtls.i1(p) + N_GHOSTS + 1,
                     prtls.i2(p) + N_GHOSTS,
                     prtls.i3(p) + N_GHOSTS,
                     cur::jx2,
                     Fx2_2 * Wx1_2 * (ONE - Wx3_2));
          deposit_at(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS,
                     prtls.i3(p) + N_GHOSTS + 1,
                     cur::jx2,
                     Fx2_2 * (ONE - Wx1_2) * Wx3_2);
          deposit_at(prtls.i1(p) + N_GHOSTS + 1,
                     prtls.i2(p) + N_GHOSTS,
                     prtls.i3(p) + N_GHOSTS + 1,
                     cur::jx2,
                     Fx2_2 * Wx1_2 * Wx3_2);

          deposit_at(prtls.i1_prev(p) + N_GHOSTS,
                     prtls.i2_prev(p) + N_GHOSTS,
                     prtls.i3_prev(p) + N_GHOSTS,
                     cur::jx3,
                     Fx3_1 * (ONE - Wx1_1) * (ONE - Wx2_1));
          deposit_at(prtls.i1_prev(p) + N_GHOSTS + 1,
                     prtls.i2_prev(p) + N_GHOSTS,
                     prtls.i3_prev(p) + N_GHOSTS,
                     cur::jx3,
                     Fx3_1 * Wx1_1 * (ONE - Wx2_1));
          deposit_at(prtls.i1_prev(p) + N_GHOSTS,
                     prtls.i2_prev(p) + N_GHOSTS + 1,
                     prtls.i3_prev(p) + N_GHOSTS,
                     cur::jx3,
                     Fx3_1 * (ONE - Wx1_1) * Wx2_1);
          deposit_at(prtls.i1_prev(p) + N_GHOSTS + 1,
                     prtls.i2_prev(p) + N_GHOSTS + 1,
                     prtls.i3_prev(p) + N_GHOSTS,
                     cur::jx3,
                     Fx3_1 * Wx1_1 * Wx2_1);

          deposit_at(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS,
                     prtls.i3(p) + N_GHOSTS,
                     cur::jx3,
                     Fx3_2 * (ONE - Wx1_2) * (ONE - Wx2_2));
          deposit_at(prtls.i1(p) + N_GHOSTS + 1,
                     prtls.i2(p) + N_GHOSTS,
                     prtls.i3(p) + N_GHOSTS,
                     cur::jx3,
                     Fx3_2 * Wx1_2 * (ONE - Wx2_2));
          deposit_at(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS + 1,
                     prtls.i3(p) + N_GHOSTS,
                     cur::jx3,
                     Fx3_2 * (ONE - Wx1_2) * Wx2_2);
          deposit_at(prtls.i1(p) + N_GHOSTS + 1,
                     prtls.i2(p) + N_GHOSTS + 1,
                     prtls.i3(p) + N_GHOSTS,
                     cur::jx3,
                     Fx3_2 * Wx1_2 * Wx2_2);
        }
      }
    } else if constexpr ((O >= 1u) and (O <= 11u)) {

      // shape function in dim1 -> always required
      real_t iS_x1[O + 2], fS_x1[O + 2];
      // indices of the shape function
      int    i1_min, i1_max;

      // call shape function
      prtl_shape::for_deposit<O>(prtls.i1_prev(p),
                                 static_cast<real_t>(prtls.dx1_prev(p)),
                                 prtls.i1(p),
                                 static_cast<real_t>(prtls.dx1(p)),
                                 i1_min,
                                 i1_max,
                                 iS_x1,
                                 fS_x1);

      if constexpr (D == Dim::_1D) {
        // (1D): fused Esirkepov, no [O+2] temporaries.
        //   jx1[i] = -Qdx1dt * sum_{i'=0}^{i} (fS_x1[i'] - iS_x1[i'])
        //          = -Qdx1dt * P1[i]                  (Eq. 38, 1D)
        //   Wx23[i] = HALF * (fS_x1[i] + iS_x1[i])     (computed inline)
        const real_t Qdx1dt = coeff * inv_dt;
        const real_t QVx2   = coeff * vp[1];
        const real_t QVx3   = coeff * vp[2];

        // account for ghost cells
        i1_min += N_GHOSTS;
        i1_max += N_GHOSTS;

        // get number of update indices for asymmetric movement
        const int di_x1 = i1_max - i1_min;

        // Current update — fused over the union line so the J cell
        // stays L1-resident across the 3 component atomic_adds.
        real_t P1 = ZERO;
        for (int i = 0; i <= di_x1; ++i) {
          P1                += fS_x1[i] - iS_x1[i];
          const int    gi    = i1_min + i;
          const real_t Wx23  = HALF * (fS_x1[i] + iS_x1[i]);
          if (i < di_x1) {
            deposit_at(gi, cur::jx1, -Qdx1dt * P1);
          }
          deposit_at(gi, cur::jx2, QVx2 * Wx23);
          deposit_at(gi, cur::jx3, QVx3 * Wx23);
        }

      } else if constexpr (D == Dim::_2D) {

        // shape function in dim1 -> always required
        real_t iS_x2[O + 2], fS_x2[O + 2];
        // indices of the shape function
        int    i2_min, i2_max;

        // call shape function
        prtl_shape::for_deposit<O>(prtls.i2_prev(p),
                                   static_cast<real_t>(prtls.dx2_prev(p)),
                                   prtls.i2(p),
                                   static_cast<real_t>(prtls.dx2(p)),
                                   i2_min,
                                   i2_max,
                                   iS_x2,
                                   fS_x2);

        /**
         * (2D): fused Esirkepov, no [O+2]^2 temporaries.
         *
         * Esirkepov 2001 Eq. 38 (simplified) is separable: with
         * P1[i] = sum_{i'=0}^{i} (fS_x1[i'] - iS_x1[i']) and
         * P2[j] = sum_{j'=0}^{j} (fS_x2[j'] - iS_x2[j']),
         *   jx1[i][j] = -Q*HALF * P1[i] * (fS_x2[j] + iS_x2[j])
         *   jx2[i][j] = -Q*HALF * P2[j] * (fS_x1[i] + iS_x1[i])
         *   Wx3[i][j] = THIRD*( fS_x2[j]*(HALF*iS_x1[i]+fS_x1[i])
         *                     + iS_x2[j]*(HALF*fS_x1[i]+iS_x1[i]) )
         * with Q = coeff*inv_dt (Qdx1dt == Qdx2dt). Same value as the
         * old explicit Wx/jx tensors up to FP reassociation;
         * charge-conserving by construction. Prefix sums carried as
         * running scalars, so the only per-thread state is the
         * existing 1D shape arrays.
         */
        const real_t QVx3 = coeff * vp[2];
        // -Q*HALF prefactor (Qdx1dt == Qdx2dt == coeff*inv_dt)
        const real_t cf   = -(coeff * inv_dt) * HALF;

        // account for ghost cells
        i1_min += N_GHOSTS;
        i2_min += N_GHOSTS;
        i1_max += N_GHOSTS;
        i2_max += N_GHOSTS;

        // get number of update indices for asymmetric movement
        const int di_x1 = i1_max - i1_min;
        const int di_x2 = i2_max - i2_min;

        // Current update — fused over the union plane so the J cell
        // line stays L1-resident across the 3 component atomic_adds.
        real_t P1 = ZERO;
        for (int i = 0; i <= di_x1; ++i) {
          P1                += fS_x1[i] - iS_x1[i];
          const int    gi    = i1_min + i;
          const real_t iSx1  = iS_x1[i];
          const real_t fSx1  = fS_x1[i];
          const real_t A1    = fSx1 + iSx1; // jx2 cross-factor
          real_t       P2    = ZERO;
          for (int j = 0; j <= di_x2; ++j) {
            P2                += fS_x2[j] - iS_x2[j];
            const int    gj    = i2_min + j;
            const real_t iSx2  = iS_x2[j];
            const real_t fSx2  = fS_x2[j];
            if (i < di_x1) {
              deposit_at(gi, gj, cur::jx1, cf * P1 * (fSx2 + iSx2));
            }
            if (j < di_x2) {
              deposit_at(gi, gj, cur::jx2, cf * P2 * A1);
            }
            const real_t Wx3 = THIRD * (fSx2 * (HALF * iSx1 + fSx1) +
                                        iSx2 * (HALF * fSx1 + iSx1));
            deposit_at(gi, gj, cur::jx3, QVx3 * Wx3);
          }
        }

      } else if constexpr (D == Dim::_3D) {
        // shape function in dim2
        real_t iS_x2[O + 2], fS_x2[O + 2];
        // indices of the shape function
        int    i2_min, i2_max;
        // call shape function
        prtl_shape::for_deposit<O>(prtls.i2_prev(p),
                                   static_cast<real_t>(prtls.dx2_prev(p)),
                                   prtls.i2(p),
                                   static_cast<real_t>(prtls.dx2(p)),
                                   i2_min,
                                   i2_max,
                                   iS_x2,
                                   fS_x2);

        // shape function in dim3
        real_t iS_x3[O + 2], fS_x3[O + 2];
        // indices of the shape function
        int    i3_min, i3_max;

        // call shape function
        prtl_shape::for_deposit<O>(prtls.i3_prev(p),
                                   static_cast<real_t>(prtls.dx3_prev(p)),
                                   prtls.i3(p),
                                   static_cast<real_t>(prtls.dx3(p)),
                                   i3_min,
                                   i3_max,
                                   iS_x3,
                                   fS_x3);

        /**
         * fused Esirkepov, no (O+2)^3 temporaries.
         *
         * The Esirkepov 3D current (2001, Eq. 31) is separable: with
         * P1[i] = sum_{i'=0}^{i} (fS_x1[i'] - iS_x1[i']) (and likewise
         * P2[j], P3[k]) the cumulative-sum currents collapse to
         *
         *   jx1[i][j][k] = -Q*THIRD * P1[i] * G23(j,k)
         *   jx2[i][j][k] = -Q*THIRD * P2[j] * H13(i,k)
         *   jx3[i][j][k] = -Q*THIRD * P3[k] * F12(i,j)
         *
         * with the 1D-shape cross-factors
         *
         *   G23(j,k) = iS_x2[j]*iS_x3[k] + fS_x2[j]*fS_x3[k]
         *            + HALF*(iS_x3[k]*fS_x2[j] + iS_x2[j]*fS_x3[k])
         *   H13(i,k) = iS_x1[i]*iS_x3[k] + fS_x1[i]*fS_x3[k]
         *            + HALF*(iS_x3[k]*fS_x1[i] + iS_x1[i]*fS_x3[k])
         *   F12(i,j) = iS_x1[i]*iS_x2[j] + fS_x1[i]*fS_x2[j]
         *            + HALF*(iS_x1[i]*fS_x2[j] + iS_x2[j]*fS_x1[i])
         *
         * and Q = coeff*inv_dt (Qdxdt == Qdydt == Qdzdt). This is the
         * same value as the old explicit Wx/jx tensors up to
         * floating-point reassociation: charge-conserving by
         * construction (the Esirkepov decomposition is exact). The
         * prefix sums are carried as running scalars in the deposit
         * loop, so the only per-thread state is the existing 1D shape
         * arrays (no (O+2)^3 / (O+2)^2 locals, hence far fewer VGPRs
         * and no private-memory tensor traffic).
         */

        // account for ghost cells
        i1_min += N_GHOSTS;
        i2_min += N_GHOSTS;
        i3_min += N_GHOSTS;
        i1_max += N_GHOSTS;
        i2_max += N_GHOSTS;
        i3_max += N_GHOSTS;

        // get number of update indices for asymmetric movement
        const int di_x1 = i1_max - i1_min;
        const int di_x2 = i2_max - i2_min;
        const int di_x3 = i3_max - i3_min;

        // -Q*THIRD prefactor (Qdxdt == Qdydt == Qdzdt == coeff*inv_dt)
        const real_t cf = -(coeff * inv_dt) * THIRD;

        /**
         * Current update — fused over the union cube so the J cell
         * line stays L1-resident across the 3 component atomic_adds.
         * Per-cell branches on (i<di_x1), (j<di_x2), (k<di_x3) skip
         * the trailing slab where each component's stencil ends one
         * cell short of the union; particles within a tile share
         * di_x* so the branch predicates cleanly.
         */
        real_t P1 = ZERO;
        for (int i = 0; i <= di_x1; ++i) {
          P1                 += fS_x1[i] - iS_x1[i];
          const int    gi     = i1_min + i;
          const real_t iSx1i  = iS_x1[i];
          const real_t fSx1i  = fS_x1[i];
          real_t       P2     = ZERO;
          for (int j = 0; j <= di_x2; ++j) {
            P2                 += fS_x2[j] - iS_x2[j];
            const int    gj     = i2_min + j;
            const real_t iSx2j  = iS_x2[j];
            const real_t fSx2j  = fS_x2[j];
            const real_t F12    = iSx1i * iSx2j + fSx1i * fSx2j +
                               HALF * (iSx1i * fSx2j + iSx2j * fSx1i);
            real_t P3 = ZERO;
            for (int k = 0; k <= di_x3; ++k) {
              P3                 += fS_x3[k] - iS_x3[k];
              const int    gk     = i3_min + k;
              const real_t iSx3k  = iS_x3[k];
              const real_t fSx3k  = fS_x3[k];
              if (i < di_x1) {
                const real_t G23 = iSx2j * iSx3k + fSx2j * fSx3k +
                                   HALF * (iSx3k * fSx2j + iSx2j * fSx3k);
                deposit_at(gi, gj, gk, cur::jx1, cf * P1 * G23);
              }
              if (j < di_x2) {
                const real_t H13 = iSx1i * iSx3k + fSx1i * fSx3k +
                                   HALF * (iSx3k * fSx1i + iSx1i * fSx3k);
                deposit_at(gi, gj, gk, cur::jx2, cf * P2 * H13);
              }
              if (k < di_x3) {
                deposit_at(gi, gj, gk, cur::jx3, cf * P3 * F12);
              }
            }
          }
        }

      } // dim
    } else { // order
      raise::KernelError(
        HERE,
        "Unsupported interpolation order. O > 11 not supported. Seriously. "
        "What are you even doing here? Entity already goes to 11!");
    }
  }

} // namespace kernel

#undef i_di_to_Xi

#endif // KERNELS_DEPOSITION_CURRENTS_SINGLE_PARTICLE_HPP
