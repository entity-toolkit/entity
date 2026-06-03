/**
 * @file kernels/currents_deposit.hpp
 * @brief Covariant algorithms for the current deposition.
 *
 * Two kernels share the same per-particle body
 * (`kernel::DepositOneParticle`):
 *   - `kernel::DepositCurrents_kernel<S, M, O>` flat (RangePolicy over particles,
 *     writes into a `Kokkos::Experimental::ScatterView`). Always available.
 *   - `kernel::DepositCurrents_kernel_tiled<S, M, O, T_TILE>` team-policy
 *     (one team per spatial tile, accumulates into team SLM scratch with
 *     atomic adds, then flushes to global J). Available when `team_policy=ON`
 *     (`#if defined(TEAM_POLICY)`). Stream 2 of the Pattern A plan.
 *
 * @implements
 *   - kernel::deposit::PrtlPack<>
 *   - kernel::DepositOneParticle<>
 *   - kernel::DepositCurrents_kernel<>
 *   - kernel::DepositCurrents_kernel_tiled<>   (TEAM_POLICY only)
 * @namespaces:
 *   - kernel::
 *   - kernel::deposit::
 */

#ifndef KERNELS_CURRENTS_DEPOSIT_HPP
#define KERNELS_CURRENTS_DEPOSIT_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "kernels/particle_shapes.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

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

  /**
   * @brief Flat current-deposition kernel.
   *
   * One thread per particle (RangePolicy). Writes are coalesced through a
   * `Kokkos::Experimental::ScatterView` to avoid per-thread atomics on
   * global J. Constructor signature is unchanged from prior versions —
   * `engines/srpic/currents.h` continues to call it identically.
   */
  template <SimEngine::type S, MetricClass M, unsigned short O = 1u>
  class DepositCurrents_kernel {
    static_assert(O <= 11u, "Shape function order O must be <= 11");
    static constexpr auto D = M::Dim;

    scatter_ndfield_t<D, 3> J;
    const ParticleArrays    prtls;
    const M                 metric;
    const real_t            charge, inv_dt;

  public:
    DepositCurrents_kernel(const scatter_ndfield_t<D, 3>& scatter_cur,
                           const ParticleArrays&          prtls,
                           const M&                       metric,
                           real_t                         charge,
                           const real_t                   dt)
      : J { scatter_cur }
      , prtls { prtls }
      , metric { metric }
      , charge { charge }
      , inv_dt { ONE / dt } {
      raise::ErrorIf(
        (O == 2u and N_GHOSTS < 2),
        "Order of interpolation is 2, but number of ghost cells is < 2",
        HERE);
    }

    Inline auto operator()(prtlidx_t p) const -> void {
      auto J_acc = J.access();
      if constexpr (D == Dim::_1D) {
        DepositOneParticle<S, M, O>(p,
                                    prtls,
                                    metric,
                                    charge,
                                    inv_dt,
                                    [&](int g_i1, int comp, real_t v) {
                                      J_acc(g_i1, comp) += v;
                                    });
      } else if constexpr (D == Dim::_2D) {
        DepositOneParticle<S, M, O>(p,
                                    prtls,
                                    metric,
                                    charge,
                                    inv_dt,
                                    [&](int g_i1, int g_i2, int comp, real_t v) {
                                      J_acc(g_i1, g_i2, comp) += v;
                                    });
      } else if constexpr (D == Dim::_3D) {
        DepositOneParticle<S, M, O>(
          p,
          prtls,
          metric,
          charge,
          inv_dt,
          [&](int g_i1, int g_i2, int g_i3, int comp, real_t v) {
            J_acc(g_i1, g_i2, g_i3, comp) += v;
          });
      }
    }
  };

  /**
   * @brief Tiled current-deposition kernel.
   *
   * One team per spatial tile (`league_size = ntiles_total`). Each team
   * accumulates particle contributions into a per-team scratch buffer of
   * shape `(T_TILE + 2*HALO)^D × 3` real_t, where `HALO = O + 1` cells per
   * side. Scratch atomics live in SLM (PVC: ~5–10 cycles per
   * `atomic_add`); the global J is touched only once per scratch cell at
   * flush time. Compared with the flat scatter-view kernel:
   *   - global atomic pressure ~ (T_TILE + 2*HALO)^D × 3 per tile
   *     instead of (stencil writes per particle × particles)
   *   - per-particle stencil writes are tile-local (SLM) instead of
   *     scattering through global HBM
   *
   * Supports `O ∈ {0, ..., 11}`. `O == 0` (zigzag) is wired for
   * A/B benchmarking against the flat scatter-view kernel — its narrow
   * stencil typically makes scratch alloc/zero/flush overhead a
   * regression there, but it's good to be able to measure the
   * crossover. To revert and use flat for zigzag-only builds, change
   * the dispatch in `engines/srpic/currents.h` from
   * `#if defined(TEAM_POLICY)` to
   * `#if defined(TEAM_POLICY) && (SHAPE_ORDER > 0)`.
   *
   * Particle iteration order is governed by `tile_offsets`: tile `t`
   * owns particles `[tile_offsets(t), tile_offsets(t+1))`, post-sort.
   * `SortSpatially` (`particles_sort.cpp`) is responsible for keeping
   * the SoA arrays consistent with that.
   *
   * **Halo sizing and escape valve.** Sort runs at the end of the
   * previous step (see `srpic.hpp`), so at deposit time the particle
   * has already been pushed once — its `min(i, i_prev)` may differ
   * from the bin key by one cell of drift per step elapsed since the
   * last sort. The scratch HALO is `STENCIL_REACH(O) + DRIFT`, where
   * `STENCIL_REACH = 2` for zigzag (writes `{i_prev, i_prev+1, i,
   * i+1}` ⇒ +2 above `min(i, i_prev)` with `|Δi|=1`) and `O` for
   * Esirkepov, and `DRIFT` is a fixed constant (1) covering the one
   * guaranteed post-sort pusher step.
   *
   * HALO is sized for the *common* (every-step-sorted) case, not for
   * a worst-case sort cadence: correctness does **not** depend on it.
   * Any particle whose stencil escapes the scratch tile — because it
   * drifted further than `DRIFT` (e.g. a large runtime
   * `spatial_sorting_interval`), or because the halo is otherwise
   * undersized — silently falls back to a direct, bounds-clipped
   * `Kokkos::atomic_add` on the global J view. That path is
   * charge-conserving (each particle's stencil is deposited exactly
   * once, partly to private SLM scratch and partly to global J, and
   * scratch is flushed once via `atomic_add`); it is merely slower
   * per write. Sorting less often than every step therefore costs
   * escape-valve traffic, never accuracy.
   */
  template <SimEngine::type S, MetricClass M, unsigned short O, unsigned short T_TILE>
  class DepositCurrentsTiled_kernel {
    static_assert(O <= 11u, "Shape order O must be <= 11");
    static_assert(T_TILE > 0u, "T_TILE must be positive");
    static constexpr auto D = M::Dim;

    /**
     * Per-side scratch halo, derived from first principles.
     *
     *   total halo = stencil_reach(O) + drift_between_sort_and_deposit
     *
     * stencil_reach(O) — maximum cells the deposit writes ABOVE
     * min(i, i_prev) under CFL |v * dt/dx| <= 1/2:
     *   - O == 0 (zigzag):  writes { i_prev, i_prev+1, i, i+1 } => +2
     *   - O >= 1 Esirkepov: `for_deposit` returns an (O+2)-wide
     *     array but only O+1 entries are non-zero, and the union
     *     window satisfies `i_max - i_min <= O+1` (see
     *     particle_shapes.hpp::for_deposit). The genuine one-sided
     *     reach above min(i, i_prev) is therefore O, not O+1 — the
     *     old `O+1` carried one extra cell of conservative padding
     *     on top of the already-conservative drift term below.
     *
     * drift — sort runs at end-of-step (see srpic.hpp), so a particle
     * sees exactly one pusher step before the *next* step's deposit
     * when sorted every step (the common case). DRIFT is therefore a
     * fixed constant of 1, NOT a compile-time function of the runtime
     * sort cadence. Sizing the halo for the common case (rather than a
     * worst-case sort interval) is what keeps the scratch small enough
     * for good occupancy; a species sorted less often than
     * every step just drifts past the halo and takes the global-J
     * escape valve more often — correct, only slower (see the class
     * doc-comment for why this is charge-conserving).
     */
    static constexpr int STENCIL_REACH = (O == 0u) ? 2 : static_cast<int>(O);
    static constexpr int DRIFT         = 1;
    static constexpr int HALO          = STENCIL_REACH + DRIFT;
    static constexpr int TE            = static_cast<int>(T_TILE) + 2 * HALO;

    using exec_space  = Kokkos::DefaultExecutionSpace;
    using team_policy = Kokkos::TeamPolicy<exec_space>;
    using member_t    = typename team_policy::member_type;

    ndfield_t<D, 3> J;
    ParticleArrays  prtls;
    const M         metric;
    const real_t    charge, inv_dt;

    // Tile metadata produced by SortSpatially.
    array_t<npart_t*> tile_offsets;
    ncells_t          ntx1 { 1u }, ntx2 { 1u }, ntx3 { 1u };
    ncells_t          total_tiles { 0u };

    /**
     * J's full storage extent including all ghost cells. Used to clip
     * the cooperative flush so that a partial tile at the high end of
     * the domain does not over-write past the J view.
     */
    int j_ext1 { 0 }, j_ext2 { 0 }, j_ext3 { 0 };

  public:
    DepositCurrentsTiled_kernel(const ndfield_t<D, 3>& cur,
                                const ParticleArrays&  prtls,
                                const M&               metric,
                                real_t                 charge,
                                real_t                 dt,
                                const TileLayout<D>&   layout)
      : J { cur }
      , prtls { prtls }
      , metric { metric }
      , charge { charge }
      , inv_dt { ONE / dt }
      , tile_offsets { layout.tile_offsets }
      , ntx1 { layout.ntiles_per_axis[0] }
      , ntx2 { layout.ntiles_per_axis[1] }
      , ntx3 { layout.ntiles_per_axis[2] }
      , total_tiles { layout.ntiles_total } {
      raise::ErrorIf(
        layout.tile_size != T_TILE,
        "Tiled deposit launched with mismatched T_TILE and runtime tile_size",
        HERE);
      /**
       * @note: HALO is allowed to exceed N_GHOSTS. The cooperative
       * scratch→J flush and the per-particle escape valve both bounds-clip
       * their writes against `j_ext*` so writes that would land past J's
       * ghost stripe are silently dropped (they only ever come from a
       * particle whose stencil reaches into the domain ghost region, where
       * CommunicateFields will re-supply the contribution).
       */
      if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
        j_ext1 = static_cast<int>(cur.extent(0));
      }
      if constexpr (D == Dim::_2D or D == Dim::_3D) {
        j_ext2 = static_cast<int>(cur.extent(1));
      }
      if constexpr (D == Dim::_3D) {
        j_ext3 = static_cast<int>(cur.extent(2));
      }
    }

    /**
     * @brief Per-team scratch size in bytes. Used by the launcher to set
     *        `team_policy.set_scratch_size(0, Kokkos::PerTeam(bytes))`.
     */
    static constexpr size_t scratch_bytes() {
      if constexpr (D == Dim::_1D) {
        return scratch_ndfield_t<D, 3, real_t>::shmem_size(TE, 3);
      } else if constexpr (D == Dim::_2D) {
        return scratch_ndfield_t<D, 3, real_t>::shmem_size(TE, TE, 3);
      } else {
        return scratch_ndfield_t<D, 3, real_t>::shmem_size(TE, TE, TE, 3);
      }
    }

    Inline void operator()(const member_t& team) const {
      const auto tile_id = static_cast<ncells_t>(team.league_rank());
      /**
       * Tile coordinates (tile-grid indices) → tile origin in **active**
       * cell coords (no ghost offset). Using ncells_t to match the linearised
       * tile index produced by SortSpatially.
       */
      ncells_t   tx1 = 0, tx2 = 0, tx3 = 0;
      if constexpr (D == Dim::_1D) {
        tx1 = tile_id;
      } else if constexpr (D == Dim::_2D) {
        tx1 = tile_id / ntx2;
        tx2 = tile_id - tx1 * ntx2;
      } else {
        const auto plane = ntx2 * ntx3;
        tx1              = tile_id / plane;
        const auto rem   = tile_id - tx1 * plane;
        tx2              = rem / ntx3;
        tx3              = rem - tx2 * ntx3;
      }
      /**
       * origin_active = lowest active-cell index in the tile (no ghost).
       * origin_J      = same value translated into J's storage coordinate
       *                 (i.e. plus N_GHOSTS).
       * origin_J_low  = J coordinate of scratch index 0 (i.e. origin_J - HALO).
       * local index `li` in scratch ↔ global J index `gi = li + origin_J_low`.
       */
      const int origin_J1_low = static_cast<int>(tx1 * T_TILE) +
                                static_cast<int>(N_GHOSTS) - HALO;
      const int origin_J2_low = static_cast<int>(tx2 * T_TILE) +
                                static_cast<int>(N_GHOSTS) - HALO;
      const int origin_J3_low = static_cast<int>(tx3 * T_TILE) +
                                static_cast<int>(N_GHOSTS) - HALO;

      // Allocate scratch and cooperatively zero-fill it.
      if constexpr (D == Dim::_1D) {
        scratch_ndfield_t<D, 3, real_t> scr { team.team_scratch(0), TE };
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TE * 3),
                             [&](ncells_t idx) {
                               const auto li = idx / 3;
                               const auto c  = idx - li * 3;
                               scr(li, c)    = ZERO;
                             });
        team.team_barrier();

        const auto p_begin = tile_offsets(tile_id);
        const auto p_end   = tile_offsets(tile_id + 1u);
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, p_begin, p_end),
          [&](prtlidx_t p) {
            DepositOneParticle<S, M, O>(
              p,
              prtls,
              metric,
              charge,
              inv_dt,
              /**
               * Escape valve: a particle whose stencil reaches past the
               * tile's scratch (e.g. exceeded the compile-time
               * STENCIL_REACH + DRIFT budget) falls back to a direct
               * atomic_add on the global J view. Bounds-clipped against
               * J's storage extent so writes past the domain ghost stripe
               * are dropped (matches the cooperative flush below; those
               * contributions are re-supplied by SynchronizeFields(J)).
               */
              [&](int g_i1, int comp, real_t v) {
                const int li = g_i1 - origin_J1_low;
                if (li >= 0 and li < TE) {
                  Kokkos::atomic_add(&scr(li, comp), v);
                } else if (g_i1 >= 0 and g_i1 < j_ext1) {
                  Kokkos::atomic_add(&J(g_i1, comp), v);
                }
              });
          });
        team.team_barrier();

        /**
         * Cooperative flush of scratch to global J. Bounds-clip against
         * the J view extent in case a partial high-end tile (or non-zero
         * halo at domain edges) would otherwise write past J.
         */
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TE * 3),
                             [&](const int idx) {
                               const auto li = idx / 3;
                               const auto c  = idx - li * 3;
                               const auto gi = li + origin_J1_low;
                               if (gi < 0 or gi >= j_ext1) {
                                 return;
                               }
                               const real_t v = scr(li, c);
                               if (v != ZERO) {
                                 Kokkos::atomic_add(&J(gi, c), v);
                               }
                             });
      } else if constexpr (D == Dim::_2D) {
        scratch_ndfield_t<D, 3, real_t> scr { team.team_scratch(0), TE, TE };
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, SQR(TE) * 3),
                             [&](const int idx) {
                               const auto lij = idx / 3;
                               const auto c   = idx - lij * 3;
                               const auto li  = lij / TE;
                               const auto lj  = lij - li * TE;
                               scr(li, lj, c) = ZERO;
                             });
        team.team_barrier();

        const auto p_begin = tile_offsets(tile_id);
        const auto p_end   = tile_offsets(tile_id + 1u);
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, p_begin, p_end),
          [&](prtlidx_t p) {
            DepositOneParticle<S, M, O>(
              p,
              prtls,
              metric,
              charge,
              inv_dt,
              // See 1D branch for rationale.
              [&](const int g_i1, const int g_i2, int comp, real_t v) {
                const auto li = g_i1 - origin_J1_low;
                const auto lj = g_i2 - origin_J2_low;
                if ((li >= 0 and li < TE) and (lj >= 0 and lj < TE)) {
                  Kokkos::atomic_add(&scr(li, lj, comp), v);
                } else if ((g_i1 >= 0 and g_i1 < j_ext1) and
                           (g_i2 >= 0 and g_i2 < j_ext2)) {
                  Kokkos::atomic_add(&J(g_i1, g_i2, comp), v);
                }
              });
          });
        team.team_barrier();

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, SQR(TE) * 3),
                             [&](const int idx) {
                               const auto lij = idx / 3;
                               const auto c   = idx - lij * 3;
                               const auto li  = lij / TE;
                               const auto lj  = lij - li * TE;
                               const auto gi  = li + origin_J1_low;
                               const auto gj  = lj + origin_J2_low;
                               if ((gi < 0 or gi >= j_ext1) or
                                   (gj < 0 or gj >= j_ext2)) {
                                 return;
                               }
                               const real_t v = scr(li, lj, c);
                               if (v != ZERO) {
                                 Kokkos::atomic_add(&J(gi, gj, c), v);
                               }
                             });
      } else if constexpr (D == Dim::_3D) {
        scratch_ndfield_t<D, 3, real_t> scr { team.team_scratch(0), TE, TE, TE };
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, CUBE(TE) * 3),
                             [&](const int idx) {
                               const auto lijk    = idx / 3;
                               const auto c       = idx - lijk * 3;
                               const auto li      = lijk / (TE * TE);
                               const auto rem     = lijk - li * TE * TE;
                               const auto lj      = rem / TE;
                               const auto lk      = rem - lj * TE;
                               scr(li, lj, lk, c) = ZERO;
                             });
        team.team_barrier();

        const auto p_begin = tile_offsets(tile_id);
        const auto p_end   = tile_offsets(tile_id + 1u);
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, p_begin, p_end),
          [&](prtlidx_t p) {
            DepositOneParticle<S, M, O>(
              p,
              prtls,
              metric,
              charge,
              inv_dt,
              // See 1D branch for rationale.
              [&](const int g_i1, const int g_i2, const int g_i3, int comp, real_t v) {
                const auto li = g_i1 - origin_J1_low;
                const auto lj = g_i2 - origin_J2_low;
                const auto lk = g_i3 - origin_J3_low;
                if ((li >= 0 and li < TE) and (lj >= 0 and lj < TE) and
                    (lk >= 0 and lk < TE)) {
                  Kokkos::atomic_add(&scr(li, lj, lk, comp), v);
                } else if ((g_i1 >= 0 and g_i1 < j_ext1) and
                           (g_i2 >= 0 and g_i2 < j_ext2) and
                           (g_i3 >= 0 and g_i3 < j_ext3)) {
                  Kokkos::atomic_add(&J(g_i1, g_i2, g_i3, comp), v);
                }
              });
          });
        team.team_barrier();

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, CUBE(TE) * 3),
                             [&](const int idx) {
                               const int lijk = idx / 3;
                               const int c    = idx - lijk * 3;
                               const int li   = lijk / (TE * TE);
                               const int rem  = lijk - li * TE * TE;
                               const int lj   = rem / TE;
                               const int lk   = rem - lj * TE;
                               const int gi   = li + origin_J1_low;
                               const int gj   = lj + origin_J2_low;
                               const int gk   = lk + origin_J3_low;
                               if ((gi < 0 or gi >= j_ext1) or
                                   (gj < 0 or gj >= j_ext2) or
                                   (gk < 0 or gk >= j_ext3)) {
                                 return;
                               }
                               const real_t v = scr(li, lj, lk, c);
                               if (v != ZERO) {
                                 Kokkos::atomic_add(&J(gi, gj, gk, c), v);
                               }
                             });
      }
    }
  };

} // namespace kernel

#undef i_di_to_Xi

#endif // KERNELS_CURRENTS_DEPOSIT_HPP
