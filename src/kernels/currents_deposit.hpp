/**
 * @file kernels/current_deposit.hpp
 * @brief Covariant algorithms for the current deposition
 * @implements
 *   - kernel::DepositCurrents_kernel<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_CURRENTS_DEPOSIT_HPP
#define KERNELS_CURRENTS_DEPOSIT_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "particle_shapes.hpp"

#include <Kokkos_Core.hpp>

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))

namespace kernel {
  using namespace ntt;

  /**
   * @brief Algorithm for the current deposition
   */
  template <SimEngine::type S, class M, unsigned short O = 1u>
  class DepositCurrents_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D = M::Dim;

    scatter_ndfield_t<D, 3>  J;
    const array_t<int*>      i1, i2, i3;
    const array_t<int*>      i1_prev, i2_prev, i3_prev;
    const array_t<prtldx_t*> dx1, dx2, dx3;
    const array_t<prtldx_t*> dx1_prev, dx2_prev, dx3_prev;
    const array_t<real_t*>   ux1, ux2, ux3;
    const array_t<real_t*>   phi;
    const array_t<real_t*>   weight;
    const array_t<short*>    tag;
    const M                  metric;
    const real_t             charge, inv_dt;

  public:
    /**
     * @brief explicit constructor.
     */
    DepositCurrents_kernel(const scatter_ndfield_t<D, 3>& scatter_cur,
                           const array_t<int*>&           i1,
                           const array_t<int*>&           i2,
                           const array_t<int*>&           i3,
                           const array_t<int*>&           i1_prev,
                           const array_t<int*>&           i2_prev,
                           const array_t<int*>&           i3_prev,
                           const array_t<prtldx_t*>&      dx1,
                           const array_t<prtldx_t*>&      dx2,
                           const array_t<prtldx_t*>&      dx3,
                           const array_t<prtldx_t*>&      dx1_prev,
                           const array_t<prtldx_t*>&      dx2_prev,
                           const array_t<prtldx_t*>&      dx3_prev,
                           const array_t<real_t*>&        ux1,
                           const array_t<real_t*>&        ux2,
                           const array_t<real_t*>&        ux3,
                           const array_t<real_t*>&        phi,
                           const array_t<real_t*>&        weight,
                           const array_t<short*>&         tag,
                           const M&                       metric,
                           real_t                         charge,
                           const real_t                   dt)
      : J { scatter_cur }
      , i1 { i1 }
      , i2 { i2 }
      , i3 { i3 }
      , i1_prev { i1_prev }
      , i2_prev { i2_prev }
      , i3_prev { i3_prev }
      , dx1 { dx1 }
      , dx2 { dx2 }
      , dx3 { dx3 }
      , dx1_prev { dx1_prev }
      , dx2_prev { dx2_prev }
      , dx3_prev { dx3_prev }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , phi { phi }
      , weight { weight }
      , tag { tag }
      , metric { metric }
      , charge { charge }
      , inv_dt { ONE / dt } {
      raise::ErrorIf(
        (O == 2u and N_GHOSTS < 2),
        "Order of interpolation is 2, but number of ghost cells is < 2",
        HERE);
    }

    /**
     * @brief Iteration of the loop over particles.
     * @param p index.
     */
    Inline auto operator()(index_t p) const -> void {
      if (tag(p) == ParticleTag::dead) {
        return;
      }
      // recover particle velocity to deposit in unsimulated direction
      vec_t<Dim::_3D> vp { ZERO };
      {
        coord_t<M::PrtlDim> xp { ZERO };
        if constexpr (D == Dim::_1D) {
          xp[0] = i_di_to_Xi(i1(p), dx1(p));
        } else if constexpr (D == Dim::_2D) {
          if constexpr (M::PrtlDim == Dim::_3D) {
            xp[0] = i_di_to_Xi(i1(p), dx1(p));
            xp[1] = i_di_to_Xi(i2(p), dx2(p));
            xp[2] = phi(p);
          } else {
            xp[0] = i_di_to_Xi(i1(p), dx1(p));
            xp[1] = i_di_to_Xi(i2(p), dx2(p));
          }
        } else {
          xp[0] = i_di_to_Xi(i1(p), dx1(p));
          xp[1] = i_di_to_Xi(i2(p), dx2(p));
          xp[2] = i_di_to_Xi(i3(p), dx3(p));
        }
        auto inv_energy { ZERO };
        if constexpr (S == SimEngine::SRPIC) {
          metric.template transform_xyz<Idx::XYZ, Idx::U>(xp,
                                                          { ux1(p), ux2(p), ux3(p) },
                                                          vp);
          inv_energy = ONE / math::sqrt(ONE + NORM_SQR(ux1(p), ux2(p), ux3(p)));
        } else {
          metric.template transform<Idx::D, Idx::U>(xp,
                                                    { ux1(p), ux2(p), ux3(p) },
                                                    vp);
          inv_energy = ONE / math::sqrt(ONE + ux1(p) * vp[0] + ux2(p) * vp[1] +
                                        ux3(p) * vp[2]);
        }
        if (Kokkos::isnan(vp[2]) || Kokkos::isinf(vp[2])) {
          vp[2] = ZERO;
        }
        vp[0] *= inv_energy;
        vp[1] *= inv_energy;
        vp[2] *= inv_energy;
      }

      const real_t coeff { weight(p) * charge };

      // ToDo: interpolation_order as parameter
      if constexpr (O == 1u) {
        /*
          Zig-zag deposit
        */

        const auto dxp_r_1 { static_cast<prtldx_t>(i1(p) == i1_prev(p)) *
                             (dx1(p) + dx1_prev(p)) *
                             static_cast<prtldx_t>(INV_2) };

        const real_t Wx1_1 { INV_2 * (dxp_r_1 + dx1_prev(p) +
                                      static_cast<real_t>(i1(p) > i1_prev(p))) };
        const real_t Wx1_2 { INV_2 * (dx1(p) + dxp_r_1 +
                                      static_cast<real_t>(
                                        static_cast<int>(i1(p) > i1_prev(p)) +
                                        i1_prev(p) - i1(p))) };
        const real_t Fx1_1 { (static_cast<real_t>(i1(p) > i1_prev(p)) +
                              dxp_r_1 - dx1_prev(p)) *
                             coeff * inv_dt };
        const real_t Fx1_2 { (static_cast<real_t>(
                                i1(p) - i1_prev(p) -
                                static_cast<int>(i1(p) > i1_prev(p))) +
                              dx1(p) - dxp_r_1) *
                             coeff * inv_dt };

        auto J_acc = J.access();

        // tuple_t<prtldx_t, D> dxp_r;
        if constexpr (D == Dim::_1D) {
          const real_t Fx2_1 { HALF * vp[1] * coeff };
          const real_t Fx2_2 { HALF * vp[1] * coeff };

          const real_t Fx3_1 { HALF * vp[2] * coeff };
          const real_t Fx3_2 { HALF * vp[2] * coeff };

          J_acc(i1_prev(p) + N_GHOSTS, cur::jx1) += Fx1_1;
          J_acc(i1(p) + N_GHOSTS, cur::jx1)      += Fx1_2;

          J_acc(i1_prev(p) + N_GHOSTS, cur::jx2)     += Fx2_1 * (ONE - Wx1_1);
          J_acc(i1_prev(p) + N_GHOSTS + 1, cur::jx2) += Fx2_1 * Wx1_1;
          J_acc(i1(p) + N_GHOSTS, cur::jx2)          += Fx2_2 * (ONE - Wx1_2);
          J_acc(i1(p) + N_GHOSTS + 1, cur::jx2)      += Fx2_2 * Wx1_2;

          J_acc(i1_prev(p) + N_GHOSTS, cur::jx3)     += Fx3_1 * (ONE - Wx1_1);
          J_acc(i1_prev(p) + N_GHOSTS + 1, cur::jx3) += Fx3_1 * Wx1_1;
          J_acc(i1(p) + N_GHOSTS, cur::jx3)          += Fx3_2 * (ONE - Wx1_2);
          J_acc(i1(p) + N_GHOSTS + 1, cur::jx3)      += Fx3_2 * Wx1_2;
        } else if constexpr (D == Dim::_2D || D == Dim::_3D) {
          const auto dxp_r_2 { static_cast<prtldx_t>(i2(p) == i2_prev(p)) *
                               (dx2(p) + dx2_prev(p)) *
                               static_cast<prtldx_t>(INV_2) };

          const real_t Wx2_1 { INV_2 * (dxp_r_2 + dx2_prev(p) +
                                        static_cast<real_t>(i2(p) > i2_prev(p))) };
          const real_t Wx2_2 { INV_2 * (dx2(p) + dxp_r_2 +
                                        static_cast<real_t>(
                                          static_cast<int>(i2(p) > i2_prev(p)) +
                                          i2_prev(p) - i2(p))) };
          const real_t Fx2_1 { (static_cast<real_t>(i2(p) > i2_prev(p)) +
                                dxp_r_2 - dx2_prev(p)) *
                               coeff * inv_dt };
          const real_t Fx2_2 { (static_cast<real_t>(
                                  i2(p) - i2_prev(p) -
                                  static_cast<int>(i2(p) > i2_prev(p))) +
                                dx2(p) - dxp_r_2) *
                               coeff * inv_dt };

          if constexpr (D == Dim::_2D) {
            const real_t Fx3_1 { HALF * vp[2] * coeff };
            const real_t Fx3_2 { HALF * vp[2] * coeff };

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx1) += Fx1_1 * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_1 * Wx2_1;
            J_acc(i1(p) + N_GHOSTS, i2(p) + N_GHOSTS, cur::jx1) += Fx1_2 *
                                                                   (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS, i2(p) + N_GHOSTS + 1, cur::jx1) += Fx1_2 * Wx2_2;

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx2) += Fx2_1 * (ONE - Wx1_1);
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx2) += Fx2_1 * Wx1_1;
            J_acc(i1(p) + N_GHOSTS, i2(p) + N_GHOSTS, cur::jx2) += Fx2_2 *
                                                                   (ONE - Wx1_2);
            J_acc(i1(p) + N_GHOSTS + 1, i2(p) + N_GHOSTS, cur::jx2) += Fx2_2 * Wx1_2;

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * (ONE - Wx1_1) * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * Wx1_2 * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  cur::jx3) += Fx3_1 * (ONE - Wx1_1) * Wx2_1;
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS + 1,
                  cur::jx3) += Fx3_1 * Wx1_1 * Wx2_1;

            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * (ONE - Wx1_2) * (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * Wx1_2 * (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS + 1,
                  cur::jx3) += Fx3_2 * (ONE - Wx1_2) * Wx2_2;
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS + 1,
                  cur::jx3) += Fx3_2 * Wx1_2 * Wx2_2;
          } else {
            const auto   dxp_r_3 { static_cast<prtldx_t>(i3(p) == i3_prev(p)) *
                                 (dx3(p) + dx3_prev(p)) *
                                 static_cast<prtldx_t>(INV_2) };
            const real_t Wx3_1 { INV_2 * (dxp_r_3 + dx3_prev(p) +
                                          static_cast<real_t>(i3(p) > i3_prev(p))) };
            const real_t Wx3_2 { INV_2 * (dx3(p) + dxp_r_3 +
                                          static_cast<real_t>(
                                            static_cast<int>(i3(p) > i3_prev(p)) +
                                            i3_prev(p) - i3(p))) };
            const real_t Fx3_1 { (static_cast<real_t>(i3(p) > i3_prev(p)) +
                                  dxp_r_3 - dx3_prev(p)) *
                                 coeff * inv_dt };
            const real_t Fx3_2 { (static_cast<real_t>(
                                    i3(p) - i3_prev(p) -
                                    static_cast<int>(i3(p) > i3_prev(p))) +
                                  dx3(p) - dxp_r_3) *
                                 coeff * inv_dt };

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx1) += Fx1_1 * (ONE - Wx2_1) * (ONE - Wx3_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx1) += Fx1_1 * Wx2_1 * (ONE - Wx3_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_1 * (ONE - Wx2_1) * Wx3_1;
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  i3_prev(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_1 * Wx2_1 * Wx3_1;

            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx1) += Fx1_2 * (ONE - Wx2_2) * (ONE - Wx3_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS + 1,
                  i3(p) + N_GHOSTS,
                  cur::jx1) += Fx1_2 * Wx2_2 * (ONE - Wx3_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_2 * (ONE - Wx2_2) * Wx3_2;
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS + 1,
                  i3(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_2 * Wx2_2 * Wx3_2;

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx2) += Fx2_1 * (ONE - Wx1_1) * (ONE - Wx3_1);
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx2) += Fx2_1 * Wx1_1 * (ONE - Wx3_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS + 1,
                  cur::jx2) += Fx2_1 * (ONE - Wx1_1) * Wx3_1;
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS + 1,
                  cur::jx2) += Fx2_1 * Wx1_1 * Wx3_1;

            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx2) += Fx2_2 * (ONE - Wx1_2) * (ONE - Wx3_2);
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx2) += Fx2_2 * Wx1_2 * (ONE - Wx3_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS + 1,
                  cur::jx2) += Fx2_2 * (ONE - Wx1_2) * Wx3_2;
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS + 1,
                  cur::jx2) += Fx2_2 * Wx1_2 * Wx3_2;

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * (ONE - Wx1_1) * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * Wx1_1 * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * (ONE - Wx1_1) * Wx2_1;
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS + 1,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * Wx1_1 * Wx2_1;

            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * (ONE - Wx1_2) * (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * Wx1_2 * (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS + 1,
                  i3(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * (ONE - Wx1_2) * Wx2_2;
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS + 1,
                  i3(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * Wx1_2 * Wx2_2;
          }
        }
      } else if constexpr (O == 2u) {
        /*
         * Higher order charge conserving current deposition based on
         * Esirkepov (2001) https://ui.adsabs.harvard.edu/abs/2001CoPhC.135..144E/abstract
         **/

        // iS -> shape function for init position
        // fS -> shape function for final position

        // shape function at integer points (one coeff is always ZERO)
        int    i1_min;
        real_t iS_x1_0, iS_x1_1, iS_x1_2, iS_x1_3;
        real_t fS_x1_0, fS_x1_1, fS_x1_2, fS_x1_3;

        // clang-format off
        prtl_shape::for_deposit_2nd(i1_prev(p), static_cast<real_t>(dx1_prev(p)),
                                    i1(p), static_cast<real_t>(dx1(p)),
                                    i1_min,
                                    iS_x1_0, iS_x1_1, iS_x1_2, iS_x1_3,
                                    fS_x1_0, fS_x1_1, fS_x1_2, fS_x1_3);
        // clang-format on

        if constexpr (D == Dim::_1D) {
          raise::KernelNotImplementedError(HERE);
        } else if constexpr (D == Dim::_2D) {

          // shape function at integer points (one coeff is always ZERO)
          int    i2_min;
          real_t iS_x2_0, iS_x2_1, iS_x2_2, iS_x2_3;
          real_t fS_x2_0, fS_x2_1, fS_x2_2, fS_x2_3;

          // clang-format off
          prtl_shape::for_deposit_2nd(i2_prev(p), static_cast<real_t>(dx2_prev(p)),
                                      i2(p), static_cast<real_t>(dx2(p)),
                                      i2_min,
                                      iS_x2_0, iS_x2_1, iS_x2_2, iS_x2_3,
                                      fS_x2_0, fS_x2_1, fS_x2_2, fS_x2_3);
          // clang-format on
          // x1-components
          const auto Wx1_00 = HALF * (fS_x1_0 - iS_x1_0) * (fS_x2_0 + iS_x2_0);
          const auto Wx1_01 = HALF * (fS_x1_0 - iS_x1_0) * (fS_x2_1 + iS_x2_1);
          const auto Wx1_02 = HALF * (fS_x1_0 - iS_x1_0) * (fS_x2_2 + iS_x2_2);
          const auto Wx1_03 = HALF * (fS_x1_0 - iS_x1_0) * (fS_x2_3 + iS_x2_3);

          const auto Wx1_10 = HALF * (fS_x1_1 - iS_x1_1) * (fS_x2_0 + iS_x2_0);
          const auto Wx1_11 = HALF * (fS_x1_1 - iS_x1_1) * (fS_x2_1 + iS_x2_1);
          const auto Wx1_12 = HALF * (fS_x1_1 - iS_x1_1) * (fS_x2_2 + iS_x2_2);
          const auto Wx1_13 = HALF * (fS_x1_1 - iS_x1_1) * (fS_x2_3 + iS_x2_3);

          const auto Wx1_20 = HALF * (fS_x1_2 - iS_x1_2) * (fS_x2_0 + iS_x2_0);
          const auto Wx1_21 = HALF * (fS_x1_2 - iS_x1_2) * (fS_x2_1 + iS_x2_1);
          const auto Wx1_22 = HALF * (fS_x1_2 - iS_x1_2) * (fS_x2_2 + iS_x2_2);
          const auto Wx1_23 = HALF * (fS_x1_2 - iS_x1_2) * (fS_x2_3 + iS_x2_3);

          const auto Wx1_30 = HALF * (fS_x1_3 - iS_x1_3) * (fS_x2_0 + iS_x2_0);
          const auto Wx1_31 = HALF * (fS_x1_3 - iS_x1_3) * (fS_x2_1 + iS_x2_1);
          const auto Wx1_32 = HALF * (fS_x1_3 - iS_x1_3) * (fS_x2_2 + iS_x2_2);
          const auto Wx1_33 = HALF * (fS_x1_3 - iS_x1_3) * (fS_x2_3 + iS_x2_3);

          // x2-components
          const auto Wx2_00 = HALF * (fS_x1_0 + iS_x1_0) * (fS_x2_0 - iS_x2_0);
          const auto Wx2_01 = HALF * (fS_x1_0 + iS_x1_0) * (fS_x2_1 - iS_x2_1);
          const auto Wx2_02 = HALF * (fS_x1_0 + iS_x1_0) * (fS_x2_2 - iS_x2_2);
          const auto Wx2_03 = HALF * (fS_x1_0 + iS_x1_0) * (fS_x2_3 - iS_x2_3);

          const auto Wx2_10 = HALF * (fS_x1_1 + iS_x1_1) * (fS_x2_0 - iS_x2_0);
          const auto Wx2_11 = HALF * (fS_x1_1 + iS_x1_1) * (fS_x2_1 - iS_x2_1);
          const auto Wx2_12 = HALF * (fS_x1_1 + iS_x1_1) * (fS_x2_2 - iS_x2_2);
          const auto Wx2_13 = HALF * (fS_x1_1 + iS_x1_1) * (fS_x2_3 - iS_x2_3);

          const auto Wx2_20 = HALF * (fS_x1_2 + iS_x1_2) * (fS_x2_0 - iS_x2_0);
          const auto Wx2_21 = HALF * (fS_x1_2 + iS_x1_2) * (fS_x2_1 - iS_x2_1);
          const auto Wx2_22 = HALF * (fS_x1_2 + iS_x1_2) * (fS_x2_2 - iS_x2_2);
          const auto Wx2_23 = HALF * (fS_x1_2 + iS_x1_2) * (fS_x2_3 - iS_x2_3);

          const auto Wx2_30 = HALF * (fS_x1_3 + iS_x1_3) * (fS_x2_0 - iS_x2_0);
          const auto Wx2_31 = HALF * (fS_x1_3 + iS_x1_3) * (fS_x2_1 - iS_x2_1);
          const auto Wx2_32 = HALF * (fS_x1_3 + iS_x1_3) * (fS_x2_2 - iS_x2_2);
          const auto Wx2_33 = HALF * (fS_x1_3 + iS_x1_3) * (fS_x2_3 - iS_x2_3);

          // x3-components
          const auto Wx3_00 = THIRD * (fS_x2_0 * (HALF * iS_x1_0 + fS_x1_0) +
                                       iS_x2_0 * (HALF * fS_x1_0 + iS_x1_0));
          const auto Wx3_01 = THIRD * (fS_x2_1 * (HALF * iS_x1_0 + fS_x1_0) +
                                       iS_x2_1 * (HALF * fS_x1_0 + iS_x1_0));
          const auto Wx3_02 = THIRD * (fS_x2_2 * (HALF * iS_x1_0 + fS_x1_0) +
                                       iS_x2_2 * (HALF * fS_x1_0 + iS_x1_0));
          const auto Wx3_03 = THIRD * (fS_x2_3 * (HALF * iS_x1_0 + fS_x1_0) +
                                       iS_x2_3 * (HALF * fS_x1_0 + iS_x1_0));

          const auto Wx3_10 = THIRD * (fS_x2_0 * (HALF * iS_x1_1 + fS_x1_1) +
                                       iS_x2_0 * (HALF * fS_x1_1 + iS_x1_1));
          const auto Wx3_11 = THIRD * (fS_x2_1 * (HALF * iS_x1_1 + fS_x1_1) +
                                       iS_x2_1 * (HALF * fS_x1_1 + iS_x1_1));
          const auto Wx3_12 = THIRD * (fS_x2_2 * (HALF * iS_x1_1 + fS_x1_1) +
                                       iS_x2_2 * (HALF * fS_x1_1 + iS_x1_1));
          const auto Wx3_13 = THIRD * (fS_x2_3 * (HALF * iS_x1_1 + fS_x1_1) +
                                       iS_x2_3 * (HALF * fS_x1_1 + iS_x1_1));

          const auto Wx3_20 = THIRD * (fS_x2_0 * (HALF * iS_x1_2 + fS_x1_2) +
                                       iS_x2_0 * (HALF * fS_x1_2 + iS_x1_2));
          const auto Wx3_21 = THIRD * (fS_x2_1 * (HALF * iS_x1_2 + fS_x1_2) +
                                       iS_x2_1 * (HALF * fS_x1_2 + iS_x1_2));
          const auto Wx3_22 = THIRD * (fS_x2_2 * (HALF * iS_x1_2 + fS_x1_2) +
                                       iS_x2_2 * (HALF * fS_x1_2 + iS_x1_2));
          const auto Wx3_23 = THIRD * (fS_x2_3 * (HALF * iS_x1_2 + fS_x1_2) +
                                       iS_x2_3 * (HALF * fS_x1_2 + iS_x1_2));

          const auto Wx3_30 = THIRD * (fS_x2_0 * (HALF * iS_x1_3 + fS_x1_3) +
                                       iS_x2_0 * (HALF * fS_x1_3 + iS_x1_3));
          const auto Wx3_31 = THIRD * (fS_x2_1 * (HALF * iS_x1_3 + fS_x1_3) +
                                       iS_x2_1 * (HALF * fS_x1_3 + iS_x1_3));
          const auto Wx3_32 = THIRD * (fS_x2_2 * (HALF * iS_x1_3 + fS_x1_3) +
                                       iS_x2_2 * (HALF * fS_x1_3 + iS_x1_3));
          const auto Wx3_33 = THIRD * (fS_x2_3 * (HALF * iS_x1_3 + fS_x1_3) +
                                       iS_x2_3 * (HALF * fS_x1_3 + iS_x1_3));

          // x1-component
          const auto jx1_00 = Wx1_00;
          const auto jx1_10 = jx1_00 + Wx1_10;
          const auto jx1_20 = jx1_10 + Wx1_20;
          const auto jx1_30 = jx1_20 + Wx1_30;

          const auto jx1_01 = Wx1_01;
          const auto jx1_11 = jx1_01 + Wx1_11;
          const auto jx1_21 = jx1_11 + Wx1_21;
          const auto jx1_31 = jx1_21 + Wx1_31;

          const auto jx1_02 = Wx1_02;
          const auto jx1_12 = jx1_02 + Wx1_12;
          const auto jx1_22 = jx1_12 + Wx1_22;
          const auto jx1_32 = jx1_22 + Wx1_32;

          const auto jx1_03 = Wx1_03;
          const auto jx1_13 = jx1_03 + Wx1_13;
          const auto jx1_23 = jx1_13 + Wx1_23;
          const auto jx1_33 = jx1_23 + Wx1_33;

          // y-component
          const auto jx2_00 = Wx2_00;
          const auto jx2_01 = jx2_00 + Wx2_01;
          const auto jx2_02 = jx2_01 + Wx2_02;
          const auto jx2_03 = jx2_02 + Wx2_03;

          const auto jx2_10 = Wx2_10;
          const auto jx2_11 = jx2_10 + Wx2_11;
          const auto jx2_12 = jx2_11 + Wx2_12;
          const auto jx2_13 = jx2_12 + Wx2_13;

          const auto jx2_20 = Wx2_20;
          const auto jx2_21 = jx2_20 + Wx2_21;
          const auto jx2_22 = jx2_21 + Wx2_22;
          const auto jx2_23 = jx2_22 + Wx2_23;

          const auto jx2_30 = Wx2_30;
          const auto jx2_31 = jx2_30 + Wx2_31;
          const auto jx2_32 = jx2_31 + Wx2_32;
          const auto jx2_33 = jx2_32 + Wx2_33;

          i1_min  += N_GHOSTS;
          i2_min  += N_GHOSTS;

          // @TODO: not sure about the signs here
          const real_t Qdx1dt = -coeff * inv_dt;
          const real_t Qdx2dt = -coeff * inv_dt;
          const real_t QVx3   = coeff * vp[2];

          auto J_acc = J.access();

          // x1-currents
          J_acc(i1_min + 0, i2_min + 0, cur::jx1) += Qdx1dt * jx1_00;
          J_acc(i1_min + 0, i2_min + 1, cur::jx1) += Qdx1dt * jx1_01;
          J_acc(i1_min + 0, i2_min + 2, cur::jx1) += Qdx1dt * jx1_02;
          J_acc(i1_min + 0, i2_min + 3, cur::jx1) += Qdx1dt * jx1_03;

          J_acc(i1_min + 1, i2_min + 0, cur::jx1) += Qdx1dt * jx1_10;
          J_acc(i1_min + 1, i2_min + 1, cur::jx1) += Qdx1dt * jx1_11;
          J_acc(i1_min + 1, i2_min + 2, cur::jx1) += Qdx1dt * jx1_12;
          J_acc(i1_min + 1, i2_min + 3, cur::jx1) += Qdx1dt * jx1_13;

          J_acc(i1_min + 2, i2_min + 0, cur::jx1) += Qdx1dt * jx1_20;
          J_acc(i1_min + 2, i2_min + 1, cur::jx1) += Qdx1dt * jx1_21;
          J_acc(i1_min + 2, i2_min + 2, cur::jx1) += Qdx1dt * jx1_22;
          J_acc(i1_min + 2, i2_min + 3, cur::jx1) += Qdx1dt * jx1_23;

          J_acc(i1_min + 3, i2_min + 0, cur::jx1) += Qdx1dt * jx1_30;
          J_acc(i1_min + 3, i2_min + 1, cur::jx1) += Qdx1dt * jx1_31;
          J_acc(i1_min + 3, i2_min + 2, cur::jx1) += Qdx1dt * jx1_32;
          J_acc(i1_min + 3, i2_min + 3, cur::jx1) += Qdx1dt * jx1_33;

          // x2-currents
          J_acc(i1_min + 0, i2_min + 0, cur::jx2) += Qdx2dt * jx2_00;
          J_acc(i1_min + 0, i2_min + 1, cur::jx2) += Qdx2dt * jx2_01;
          J_acc(i1_min + 0, i2_min + 2, cur::jx2) += Qdx2dt * jx2_02;
          J_acc(i1_min + 0, i2_min + 3, cur::jx2) += Qdx2dt * jx2_03;

          J_acc(i1_min + 1, i2_min + 0, cur::jx2) += Qdx2dt * jx2_10;
          J_acc(i1_min + 1, i2_min + 1, cur::jx2) += Qdx2dt * jx2_11;
          J_acc(i1_min + 1, i2_min + 2, cur::jx2) += Qdx2dt * jx2_12;
          J_acc(i1_min + 1, i2_min + 3, cur::jx2) += Qdx2dt * jx2_13;

          J_acc(i1_min + 2, i2_min + 0, cur::jx2) += Qdx2dt * jx2_20;
          J_acc(i1_min + 2, i2_min + 1, cur::jx2) += Qdx2dt * jx2_21;
          J_acc(i1_min + 2, i2_min + 2, cur::jx2) += Qdx2dt * jx2_22;
          J_acc(i1_min + 2, i2_min + 3, cur::jx2) += Qdx2dt * jx2_23;

          J_acc(i1_min + 3, i2_min + 0, cur::jx2) += Qdx2dt * jx2_30;
          J_acc(i1_min + 3, i2_min + 1, cur::jx2) += Qdx2dt * jx2_31;
          J_acc(i1_min + 3, i2_min + 2, cur::jx2) += Qdx2dt * jx2_32;
          J_acc(i1_min + 3, i2_min + 3, cur::jx2) += Qdx2dt * jx2_33;

          // x3-currents
          J_acc(i1_min + 0, i2_min + 0, cur::jx3) += QVx3 * Wx3_00;
          J_acc(i1_min + 0, i2_min + 1, cur::jx3) += QVx3 * Wx3_01;
          J_acc(i1_min + 0, i2_min + 2, cur::jx3) += QVx3 * Wx3_02;
          J_acc(i1_min + 0, i2_min + 3, cur::jx3) += QVx3 * Wx3_03;

          J_acc(i1_min + 1, i2_min + 0, cur::jx3) += QVx3 * Wx3_10;
          J_acc(i1_min + 1, i2_min + 1, cur::jx3) += QVx3 * Wx3_11;
          J_acc(i1_min + 1, i2_min + 2, cur::jx3) += QVx3 * Wx3_12;
          J_acc(i1_min + 1, i2_min + 3, cur::jx3) += QVx3 * Wx3_13;

          J_acc(i1_min + 2, i2_min + 0, cur::jx3) += QVx3 * Wx3_20;
          J_acc(i1_min + 2, i2_min + 1, cur::jx3) += QVx3 * Wx3_21;
          J_acc(i1_min + 2, i2_min + 2, cur::jx3) += QVx3 * Wx3_22;
          J_acc(i1_min + 2, i2_min + 3, cur::jx3) += QVx3 * Wx3_23;

          J_acc(i1_min + 3, i2_min + 0, cur::jx3) += QVx3 * Wx3_30;
          J_acc(i1_min + 3, i2_min + 1, cur::jx3) += QVx3 * Wx3_31;
          J_acc(i1_min + 3, i2_min + 2, cur::jx3) += QVx3 * Wx3_32;
          J_acc(i1_min + 3, i2_min + 3, cur::jx3) += QVx3 * Wx3_33;

        } else if constexpr (D == Dim::_3D) {
          //   /*
          //     y - direction
          //   */
          //
          //   // shape function at previous timestep
          //   real_t   S0y_0, S0y_1, S0y_2, S0y_3;
          //   // shape function at current timestep
          //   real_t   S1y_0, S1y_1, S1y_2, S1y_3;
          //   // indices of the shape function
          //   ncells_t iy_min;
          //   bool     update_y2;
          //   // find indices and define shape function
          //   // clang-format off
          //   shape_function_2nd(S0y_0, S0y_1, S0y_2, S0y_3,
          //                      S1y_0, S1y_1, S1y_2, S1y_3,
          //                      iy_min, update_y2,
          //                      i2(p), dx2(p),
          //                      i2_prev(p), dx2_prev(p));
          //   // clang-format on
          //
          //   /*
          //     y - direction
          //   */
          //
          //   // shape function at previous timestep
          //   real_t   S0z_0, S0z_1, S0z_2, S0z_3;
          //   // shape function at current timestep
          //   real_t   S1z_0, S1z_1, S1z_2, S1z_3;
          //   // indices of the shape function
          //   ncells_t iz_min;
          //   bool     update_z2;
          //   // find indices and define shape function
          //   // clang-format off
          //   shape_function_2nd(S0z_0, S0z_1, S0z_2, S0z_3,
          //                      S1z_0, S1z_1, S1z_2, S1z_3,
          //                      iz_min, update_z2,
          //                      i3(p), dx3(p),
          //                      i3_prev(p), dx3_prev(p));
          //   // clang-format on
          //
          //   // Unrolled calculations for Wx, Wy, and Wz
          //   // clang-format off
          //   const auto Wx_0_0_0 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_0 * S0z_0 + S1y_0 * S1z_0) +
          //                          HALF * (S0z_0 * S1y_0 + S0y_0 * S1z_0));
          //   const auto Wx_0_0_1 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_0 * S0z_1 + S1y_0 * S1z_1) +
          //                          HALF * (S0z_1 * S1y_0 + S0y_0 * S1z_1));
          //   const auto Wx_0_0_2 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_0 * S0z_2 + S1y_0 * S1z_2) +
          //                          HALF * (S0z_2 * S1y_0 + S0y_0 * S1z_2));
          //   const auto Wx_0_0_3 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_0 * S0z_3 + S1y_0 * S1z_3) +
          //                          HALF * (S0z_3 * S1y_0 + S0y_0 * S1z_3));
          //
          //   const auto Wx_0_1_0 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_1 * S0z_0 + S1y_1 * S1z_0) +
          //                          HALF * (S0z_0 * S1y_1 + S0y_1 * S1z_0));
          //   const auto Wx_0_1_1 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_1 * S0z_1 + S1y_1 * S1z_1) +
          //                          HALF * (S0z_1 * S1y_1 + S0y_1 * S1z_1));
          //   const auto Wx_0_1_2 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_1 * S0z_2 + S1y_1 * S1z_2) +
          //                          HALF * (S0z_2 * S1y_1 + S0y_1 * S1z_2));
          //   const auto Wx_0_1_3 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_1 * S0z_3 + S1y_1 * S1z_3) +
          //                          HALF * (S0z_3 * S1y_1 + S0y_1 * S1z_3));
          //
          //   const auto Wx_0_2_0 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_2 * S0z_0 + S1y_2 * S1z_0) +
          //                          HALF * (S0z_0 * S1y_2 + S0y_2 * S1z_0));
          //   const auto Wx_0_2_1 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_2 * S0z_1 + S1y_2 * S1z_1) +
          //                          HALF * (S0z_1 * S1y_2 + S0y_2 * S1z_1));
          //   const auto Wx_0_2_2 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_2 * S0z_2 + S1y_2 * S1z_2) +
          //                          HALF * (S0z_2 * S1y_2 + S0y_2 * S1z_2));
          //   const auto Wx_0_2_3 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_2 * S0z_3 + S1y_2 * S1z_3) +
          //                          HALF * (S0z_3 * S1y_2 + S0y_2 * S1z_3));
          //
          //   const auto Wx_0_3_0 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_3 * S0z_0 + S1y_3 * S1z_0) +
          //                          HALF * (S0z_0 * S1y_3 + S0y_3 * S1z_0));
          //   const auto Wx_0_3_1 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_3 * S0z_1 + S1y_3 * S1z_1) +
          //                          HALF * (S0z_1 * S1y_3 + S0y_3 * S1z_1));
          //   const auto Wx_0_3_2 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_3 * S0z_2 + S1y_3 * S1z_2) +
          //                          HALF * (S0z_2 * S1y_3 + S0y_3 * S1z_2));
          //   const auto Wx_0_3_3 = THIRD * (S1x_0 - S0x_0) *
          //                         ((S0y_3 * S0z_3 + S1y_3 * S1z_3) +
          //                          HALF * (S0z_3 * S1y_3 + S0y_3 * S1z_3));
          //
          //   const auto Wx_1_0_0 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_0 * S0z_0 + S1y_0 * S1z_0) +
          //                          HALF * (S0z_0 * S1y_0 + S0y_0 * S1z_0));
          //   const auto Wx_1_0_1 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_0 * S0z_1 + S1y_0 * S1z_1) +
          //                          HALF * (S0z_1 * S1y_0 + S0y_0 * S1z_1));
          //   const auto Wx_1_0_2 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_0 * S0z_2 + S1y_0 * S1z_2) +
          //                          HALF * (S0z_2 * S1y_0 + S0y_0 * S1z_2));
          //   const auto Wx_1_0_3 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_0 * S0z_3 + S1y_0 * S1z_3) +
          //                          HALF * (S0z_3 * S1y_0 + S0y_0 * S1z_3));
          //
          //   const auto Wx_1_1_0 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_1 * S0z_0 + S1y_1 * S1z_0) +
          //                          HALF * (S0z_0 * S1y_1 + S0y_1 * S1z_0));
          //   const auto Wx_1_1_1 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_1 * S0z_1 + S1y_1 * S1z_1) +
          //                          HALF * (S0z_1 * S1y_1 + S0y_1 * S1z_1));
          //   const auto Wx_1_1_2 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_1 * S0z_2 + S1y_1 * S1z_2) +
          //                          HALF * (S0z_2 * S1y_1 + S0y_1 * S1z_2));
          //   const auto Wx_1_1_3 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_1 * S0z_3 + S1y_1 * S1z_3) +
          //                          HALF * (S0z_3 * S1y_1 + S0y_1 * S1z_3));
          //
          //   const auto Wx_1_2_0 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_2 * S0z_0 + S1y_2 * S1z_0) +
          //                          HALF * (S0z_0 * S1y_2 + S0y_2 * S1z_0));
          //   const auto Wx_1_2_1 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_2 * S0z_1 + S1y_2 * S1z_1) +
          //                          HALF * (S0z_1 * S1y_2 + S0y_2 * S1z_1));
          //   const auto Wx_1_2_2 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_2 * S0z_2 + S1y_2 * S1z_2) +
          //                          HALF * (S0z_2 * S1y_2 + S0y_2 * S1z_2));
          //   const auto Wx_1_2_3 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_2 * S0z_3 + S1y_2 * S1z_3) +
          //                          HALF * (S0z_3 * S1y_2 + S0y_2 * S1z_3));
          //
          //   const auto Wx_1_3_0 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_3 * S0z_0 + S1y_3 * S1z_0) +
          //                          HALF * (S0z_0 * S1y_3 + S0y_3 * S1z_0));
          //   const auto Wx_1_3_1 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_3 * S0z_1 + S1y_3 * S1z_1) +
          //                          HALF * (S0z_1 * S1y_3 + S0y_3 * S1z_1));
          //   const auto Wx_1_3_2 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_3 * S0z_2 + S1y_3 * S1z_2) +
          //                          HALF * (S0z_2 * S1y_3 + S0y_3 * S1z_2));
          //   const auto Wx_1_3_3 = THIRD * (S1x_1 - S0x_1) *
          //                         ((S0y_3 * S0z_3 + S1y_3 * S1z_3) +
          //                          HALF * (S0z_3 * S1y_3 + S0y_3 * S1z_3));
          //
          //   const auto Wx_2_0_0 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_0 * S0z_0 + S1y_0 * S1z_0) +
          //                          HALF * (S0z_0 * S1y_0 + S0y_0 * S1z_0));
          //   const auto Wx_2_0_1 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_0 * S0z_1 + S1y_0 * S1z_1) +
          //                          HALF * (S0z_1 * S1y_0 + S0y_0 * S1z_1));
          //   const auto Wx_2_0_2 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_0 * S0z_2 + S1y_0 * S1z_2) +
          //                          HALF * (S0z_2 * S1y_0 + S0y_0 * S1z_2));
          //   const auto Wx_2_0_3 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_0 * S0z_3 + S1y_0 * S1z_3) +
          //                          HALF * (S0z_3 * S1y_0 + S0y_0 * S1z_3));
          //
          //   const auto Wx_2_1_0 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_1 * S0z_0 + S1y_1 * S1z_0) +
          //                          HALF * (S0z_0 * S1y_1 + S0y_1 * S1z_0));
          //   const auto Wx_2_1_1 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_1 * S0z_1 + S1y_1 * S1z_1) +
          //                          HALF * (S0z_1 * S1y_1 + S0y_1 * S1z_1));
          //   const auto Wx_2_1_2 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_1 * S0z_2 + S1y_1 * S1z_2) +
          //                          HALF * (S0z_2 * S1y_1 + S0y_1 * S1z_2));
          //   const auto Wx_2_1_3 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_1 * S0z_3 + S1y_1 * S1z_3) +
          //                          HALF * (S0z_3 * S1y_1 + S0y_1 * S1z_3));
          //
          //   const auto Wx_2_2_0 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_2 * S0z_0 + S1y_2 * S1z_0) +
          //                          HALF * (S0z_0 * S1y_2 + S0y_2 * S1z_0));
          //   const auto Wx_2_2_1 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_2 * S0z_1 + S1y_2 * S1z_1) +
          //                          HALF * (S0z_1 * S1y_2 + S0y_2 * S1z_1));
          //   const auto Wx_2_2_2 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_2 * S0z_2 + S1y_2 * S1z_2) +
          //                          HALF * (S0z_2 * S1y_2 + S0y_2 * S1z_2));
          //   const auto Wx_2_2_3 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_2 * S0z_3 + S1y_2 * S1z_3) +
          //                          HALF * (S0z_3 * S1y_2 + S0y_2 * S1z_3));
          //
          //   const auto Wx_2_3_0 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_3 * S0z_0 + S1y_3 * S1z_0) +
          //                          HALF * (S0z_0 * S1y_3 + S0y_3 * S1z_0));
          //   const auto Wx_2_3_1 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_3 * S0z_1 + S1y_3 * S1z_1) +
          //                          HALF * (S0z_1 * S1y_3 + S0y_3 * S1z_1));
          //   const auto Wx_2_3_2 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_3 * S0z_2 + S1y_3 * S1z_2) +
          //                          HALF * (S0z_2 * S1y_3 + S0y_3 * S1z_2));
          //   const auto Wx_2_3_3 = THIRD * (S1x_2 - S0x_2) *
          //                         ((S0y_3 * S0z_3 + S1y_3 * S1z_3) +
          //                          HALF * (S0z_3 * S1y_3 + S0y_3 * S1z_3));
          //
          //   const real_t Qdxdt = coeff * inv_dt;
          //
          //   const auto jx_0_0_0 =          - Qdxdt * Wx_0_0_0;
          //   const auto jx_1_0_0 = jx_0_0_0 - Qdxdt * Wx_1_0_0;
          //   const auto jx_2_0_0 = jx_1_0_0 - Qdxdt * Wx_2_0_0;
          //   const auto jx_0_1_0 =          - Qdxdt * Wx_0_1_0;
          //   const auto jx_1_1_0 = jx_0_1_0 - Qdxdt * Wx_1_1_0;
          //   const auto jx_2_1_0 = jx_1_1_0 - Qdxdt * Wx_2_1_0;
          //   const auto jx_0_2_0 =          - Qdxdt * Wx_0_2_0;
          //   const auto jx_1_2_0 = jx_0_2_0 - Qdxdt * Wx_1_2_0;
          //   const auto jx_2_2_0 = jx_1_2_0 - Qdxdt * Wx_2_2_0;
          //   const auto jx_0_3_0 =          - Qdxdt * Wx_0_3_0;
          //   const auto jx_1_3_0 = jx_0_3_0 - Qdxdt * Wx_1_3_0;
          //   const auto jx_2_3_0 = jx_1_3_0 - Qdxdt * Wx_2_3_0;
          //
          //   const auto jx_0_0_1 =          - Qdxdt * Wx_0_0_1;
          //   const auto jx_1_0_1 = jx_0_0_1 - Qdxdt * Wx_1_0_1;
          //   const auto jx_2_0_1 = jx_1_0_1 - Qdxdt * Wx_2_0_1;
          //   const auto jx_0_1_1 =          - Qdxdt * Wx_0_1_1;
          //   const auto jx_1_1_1 = jx_0_1_1 - Qdxdt * Wx_1_1_1;
          //   const auto jx_2_1_1 = jx_1_1_1 - Qdxdt * Wx_2_1_1;
          //   const auto jx_0_2_1 =          - Qdxdt * Wx_0_2_1;
          //   const auto jx_1_2_1 = jx_0_2_1 - Qdxdt * Wx_1_2_1;
          //   const auto jx_2_2_1 = jx_1_2_1 - Qdxdt * Wx_2_2_1;
          //   const auto jx_0_3_1 =          - Qdxdt * Wx_0_3_1;
          //   const auto jx_1_3_1 = jx_0_3_1 - Qdxdt * Wx_1_3_1;
          //   const auto jx_2_3_1 = jx_1_3_1 - Qdxdt * Wx_2_3_1;
          //
          //   const auto jx_0_0_2 =          - Qdxdt * Wx_0_0_2;
          //   const auto jx_1_0_2 = jx_0_0_2 - Qdxdt * Wx_1_0_2;
          //   const auto jx_2_0_2 = jx_1_0_2 - Qdxdt * Wx_2_0_2;
          //   const auto jx_0_1_2 =          - Qdxdt * Wx_0_1_2;
          //   const auto jx_1_1_2 = jx_0_1_2 - Qdxdt * Wx_1_1_2;
          //   const auto jx_2_1_2 = jx_1_1_2 - Qdxdt * Wx_2_1_2;
          //   const auto jx_0_2_2 =          - Qdxdt * Wx_0_2_2;
          //   const auto jx_1_2_2 = jx_0_2_2 - Qdxdt * Wx_1_2_2;
          //   const auto jx_2_2_2 = jx_1_2_2 - Qdxdt * Wx_2_2_2;
          //   const auto jx_0_3_2 =          - Qdxdt * Wx_0_3_2;
          //   const auto jx_1_3_2 = jx_0_3_2 - Qdxdt * Wx_1_3_2;
          //   const auto jx_2_3_2 = jx_1_3_2 - Qdxdt * Wx_2_3_2;
          //
          //   const auto jx_0_0_3 =          - Qdxdt * Wx_0_0_3;
          //   const auto jx_1_0_3 = jx_0_0_3 - Qdxdt * Wx_1_0_3;
          //   const auto jx_2_0_3 = jx_1_0_3 - Qdxdt * Wx_2_0_3;
          //   const auto jx_0_1_3 =          - Qdxdt * Wx_0_1_3;
          //   const auto jx_1_1_3 = jx_0_1_3 - Qdxdt * Wx_1_1_3;
          //   const auto jx_2_1_3 = jx_1_1_3 - Qdxdt * Wx_2_1_3;
          //   const auto jx_0_2_3 =          - Qdxdt * Wx_0_2_3;
          //   const auto jx_1_2_3 = jx_0_2_3 - Qdxdt * Wx_1_2_3;
          //   const auto jx_2_2_3 = jx_1_2_3 - Qdxdt * Wx_2_2_3;
          //   const auto jx_0_3_3 =          - Qdxdt * Wx_0_3_3;
          //   const auto jx_1_3_3 = jx_0_3_3 - Qdxdt * Wx_1_3_3;
          //   const auto jx_2_3_3 = jx_1_3_3 - Qdxdt * Wx_2_3_3;
          //
          //   /*
          //     y-component
          //   */
          //   const auto Wy_0_0_0 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_0 * S0z_0 + S1x_0 * S1z_0 +
          //                          HALF * (S0z_0 * S1x_0 + S0x_0 * S1z_0));
          //   const auto Wy_0_0_1 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_0 * S0z_1 + S1x_0 * S1z_1 +
          //                          HALF * (S0z_1 * S1x_0 + S0x_0 * S1z_1));
          //   const auto Wy_0_0_2 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_0 * S0z_2 + S1x_0 * S1z_2 +
          //                          HALF * (S0z_2 * S1x_0 + S0x_0 * S1z_2));
          //   const auto Wy_0_0_3 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_0 * S0z_3 + S1x_0 * S1z_3 +
          //                          HALF * (S0z_3 * S1x_0 + S0x_0 * S1z_3));
          //
          //   const auto Wy_0_1_0 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_0 * S0z_0 + S1x_0 * S1z_0 +
          //                          HALF * (S0z_0 * S1x_0 + S0x_0 * S1z_0));
          //   const auto Wy_0_1_1 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_0 * S0z_1 + S1x_0 * S1z_1 +
          //                          HALF * (S0z_1 * S1x_0 + S0x_0 * S1z_1));
          //   const auto Wy_0_1_2 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_0 * S0z_2 + S1x_0 * S1z_2 +
          //                          HALF * (S0z_2 * S1x_0 + S0x_0 * S1z_2));
          //   const auto Wy_0_1_3 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_0 * S0z_3 + S1x_0 * S1z_3 +
          //                          HALF * (S0z_3 * S1x_0 + S0x_0 * S1z_3));
          //
          //   const auto Wy_0_2_0 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_0 * S0z_0 + S1x_0 * S1z_0 +
          //                          HALF * (S0z_0 * S1x_0 + S0x_0 * S1z_0));
          //   const auto Wy_0_2_1 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_0 * S0z_1 + S1x_0 * S1z_1 +
          //                          HALF * (S0z_1 * S1x_0 + S0x_0 * S1z_1));
          //   const auto Wy_0_2_2 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_0 * S0z_2 + S1x_0 * S1z_2 +
          //                          HALF * (S0z_2 * S1x_0 + S0x_0 * S1z_2));
          //   const auto Wy_0_2_3 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_0 * S0z_3 + S1x_0 * S1z_3 +
          //                          HALF * (S0z_3 * S1x_0 + S0x_0 * S1z_3));
          //
          //   const auto Wy_1_0_0 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_1 * S0z_0 + S1x_1 * S1z_0 +
          //                          HALF * (S0z_0 * S1x_1 + S0x_1 * S1z_0));
          //   const auto Wy_1_0_1 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_1 * S0z_1 + S1x_1 * S1z_1 +
          //                          HALF * (S0z_1 * S1x_1 + S0x_1 * S1z_1));
          //   const auto Wy_1_0_2 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_1 * S0z_2 + S1x_1 * S1z_2 +
          //                          HALF * (S0z_2 * S1x_1 + S0x_1 * S1z_2));
          //   const auto Wy_1_0_3 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_1 * S0z_3 + S1x_1 * S1z_3 +
          //                          HALF * (S0z_3 * S1x_1 + S0x_1 * S1z_3));
          //
          //   const auto Wy_1_1_0 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_1 * S0z_0 + S1x_1 * S1z_0 +
          //                          HALF * (S0z_0 * S1x_1 + S0x_1 * S1z_0));
          //   const auto Wy_1_1_1 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_1 * S0z_1 + S1x_1 * S1z_1 +
          //                          HALF * (S0z_1 * S1x_1 + S0x_1 * S1z_1));
          //   const auto Wy_1_1_2 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_1 * S0z_2 + S1x_1 * S1z_2 +
          //                          HALF * (S0z_2 * S1x_1 + S0x_1 * S1z_2));
          //   const auto Wy_1_1_3 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_1 * S0z_3 + S1x_1 * S1z_3 +
          //                          HALF * (S0z_3 * S1x_1 + S0x_1 * S1z_3));
          //
          //   const auto Wy_1_2_0 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_1 * S0z_0 + S1x_1 * S1z_0 +
          //                          HALF * (S0z_0 * S1x_1 + S0x_1 * S1z_0));
          //   const auto Wy_1_2_1 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_1 * S0z_1 + S1x_1 * S1z_1 +
          //                          HALF * (S0z_1 * S1x_1 + S0x_1 * S1z_1));
          //   const auto Wy_1_2_2 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_1 * S0z_2 + S1x_1 * S1z_2 +
          //                          HALF * (S0z_2 * S1x_1 + S0x_1 * S1z_2));
          //   const auto Wy_1_2_3 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_1 * S0z_3 + S1x_1 * S1z_3 +
          //                          HALF * (S0z_3 * S1x_1 + S0x_1 * S1z_3));
          //
          //   const auto Wy_2_0_0 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_2 * S0z_0 + S1x_2 * S1z_0 +
          //                          HALF * (S0z_0 * S1x_2 + S0x_2 * S1z_0));
          //   const auto Wy_2_0_1 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_2 * S0z_1 + S1x_2 * S1z_1 +
          //                          HALF * (S0z_1 * S1x_2 + S0x_2 * S1z_1));
          //   const auto Wy_2_0_2 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_2 * S0z_2 + S1x_2 * S1z_2 +
          //                          HALF * (S0z_2 * S1x_2 + S0x_2 * S1z_2));
          //   const auto Wy_2_0_3 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_2 * S0z_3 + S1x_2 * S1z_3 +
          //                          HALF * (S0z_3 * S1x_2 + S0x_2 * S1z_3));
          //
          //   const auto Wy_2_1_0 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_2 * S0z_0 + S1x_2 * S1z_0 +
          //                          HALF * (S0z_0 * S1x_2 + S0x_2 * S1z_0));
          //   const auto Wy_2_1_1 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_2 * S0z_1 + S1x_2 * S1z_1 +
          //                          HALF * (S0z_1 * S1x_2 + S0x_2 * S1z_1));
          //   const auto Wy_2_1_2 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_2 * S0z_2 + S1x_2 * S1z_2 +
          //                          HALF * (S0z_2 * S1x_2 + S0x_2 * S1z_2));
          //   const auto Wy_2_1_3 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_2 * S0z_3 + S1x_2 * S1z_3 +
          //                          HALF * (S0z_3 * S1x_2 + S0x_2 * S1z_3));
          //
          //   const auto Wy_2_2_0 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_2 * S0z_0 + S1x_2 * S1z_0 +
          //                          HALF * (S0z_0 * S1x_2 + S0x_2 * S1z_0));
          //   const auto Wy_2_2_1 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_2 * S0z_1 + S1x_2 * S1z_1 +
          //                          HALF * (S0z_1 * S1x_2 + S0x_2 * S1z_1));
          //   const auto Wy_2_2_2 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_2 * S0z_2 + S1x_2 * S1z_2 +
          //                          HALF * (S0z_2 * S1x_2 + S0x_2 * S1z_2));
          //   const auto Wy_2_2_3 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_2 * S0z_3 + S1x_2 * S1z_3 +
          //                          HALF * (S0z_3 * S1x_2 + S0x_2 * S1z_3));
          //
          //   const auto Wy_3_0_0 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_3 * S0z_0 + S1x_3 * S1z_0 +
          //                          HALF * (S0z_0 * S1x_3 + S0x_3 * S1z_0));
          //   const auto Wy_3_0_1 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_3 * S0z_1 + S1x_3 * S1z_1 +
          //                          HALF * (S0z_1 * S1x_3 + S0x_3 * S1z_1));
          //   const auto Wy_3_0_2 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_3 * S0z_2 + S1x_3 * S1z_2 +
          //                          HALF * (S0z_2 * S1x_3 + S0x_3 * S1z_2));
          //   const auto Wy_3_0_3 = THIRD * (S1y_0 - S0y_0) *
          //                         (S0x_3 * S0z_3 + S1x_3 * S1z_3 +
          //                          HALF * (S0z_3 * S1x_3 + S0x_3 * S1z_3));
          //
          //   const auto Wy_3_1_0 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_3 * S0z_0 + S1x_3 * S1z_0 +
          //                          HALF * (S0z_0 * S1x_3 + S0x_3 * S1z_0));
          //   const auto Wy_3_1_1 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_3 * S0z_1 + S1x_3 * S1z_1 +
          //                          HALF * (S0z_1 * S1x_3 + S0x_3 * S1z_1));
          //   const auto Wy_3_1_2 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_3 * S0z_2 + S1x_3 * S1z_2 +
          //                          HALF * (S0z_2 * S1x_3 + S0x_3 * S1z_2));
          //   const auto Wy_3_1_3 = THIRD * (S1y_1 - S0y_1) *
          //                         (S0x_3 * S0z_3 + S1x_3 * S1z_3 +
          //                          HALF * (S0z_3 * S1x_3 + S0x_3 * S1z_3));
          //
          //   const auto Wy_3_2_0 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_3 * S0z_0 + S1x_3 * S1z_0 +
          //                          HALF * (S0z_0 * S1x_3 + S0x_3 * S1z_0));
          //   const auto Wy_3_2_1 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_3 * S0z_1 + S1x_3 * S1z_1 +
          //                          HALF * (S0z_1 * S1x_3 + S0x_3 * S1z_1));
          //   const auto Wy_3_2_2 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_3 * S0z_2 + S1x_3 * S1z_2 +
          //                          HALF * (S0z_2 * S1x_3 + S0x_3 * S1z_2));
          //   const auto Wy_3_2_3 = THIRD * (S1y_2 - S0y_2) *
          //                         (S0x_3 * S0z_3 + S1x_3 * S1z_3 +
          //                          HALF * (S0z_3 * S1x_3 + S0x_3 * S1z_3));
          //
          //   const real_t Qdydt = coeff * inv_dt;
          //
          //   const auto jy_0_0_0 =          - Qdydt * Wy_0_0_0;
          //   const auto jy_0_1_0 = jy_0_0_0 - Qdydt * Wy_0_1_0;
          //   const auto jy_0_2_0 = jy_0_1_0 - Qdydt * Wy_0_2_0;
          //   const auto jy_1_0_0 =          - Qdydt * Wy_1_0_0;
          //   const auto jy_1_1_0 = jy_1_0_0 - Qdydt * Wy_1_1_0;
          //   const auto jy_1_2_0 = jy_1_1_0 - Qdydt * Wy_1_2_0;
          //   const auto jy_2_0_0 =          - Qdydt * Wy_2_0_0;
          //   const auto jy_2_1_0 = jy_2_0_0 - Qdydt * Wy_2_1_0;
          //   const auto jy_2_2_0 = jy_2_1_0 - Qdydt * Wy_2_2_0;
          //   const auto jy_3_0_0 =          - Qdydt * Wy_3_0_0;
          //   const auto jy_3_1_0 = jy_3_0_0 - Qdydt * Wy_3_1_0;
          //   const auto jy_3_2_0 = jy_3_1_0 - Qdydt * Wy_3_2_0;
          //
          //   const auto jy_0_0_1 =          - Qdydt * Wy_0_0_1;
          //   const auto jy_0_1_1 = jy_0_0_1 - Qdydt * Wy_0_1_1;
          //   const auto jy_0_2_1 = jy_0_1_1 - Qdydt * Wy_0_2_1;
          //   const auto jy_1_0_1 =          - Qdydt * Wy_1_0_1;
          //   const auto jy_1_1_1 = jy_1_0_1 - Qdydt * Wy_1_1_1;
          //   const auto jy_1_2_1 = jy_1_1_1 - Qdydt * Wy_1_2_1;
          //   const auto jy_2_0_1 =          - Qdydt * Wy_2_0_1;
          //   const auto jy_2_1_1 = jy_2_0_1 - Qdydt * Wy_2_1_1;
          //   const auto jy_2_2_1 = jy_2_1_1 - Qdydt * Wy_2_2_1;
          //   const auto jy_3_0_1 =          - Qdydt * Wy_3_0_1;
          //   const auto jy_3_1_1 = jy_3_0_1 - Qdydt * Wy_3_1_1;
          //   const auto jy_3_2_1 = jy_3_1_1 - Qdydt * Wy_3_2_1;
          //
          //   const auto jy_0_0_2 =          - Qdydt * Wy_0_0_2;
          //   const auto jy_0_1_2 = jy_0_0_2 - Qdydt * Wy_0_1_2;
          //   const auto jy_0_2_2 = jy_0_1_2 - Qdydt * Wy_0_2_2;
          //   const auto jy_1_0_2 =          - Qdydt * Wy_1_0_2;
          //   const auto jy_1_1_2 = jy_1_0_2 - Qdydt * Wy_1_1_2;
          //   const auto jy_1_2_2 = jy_1_1_2 - Qdydt * Wy_1_2_2;
          //   const auto jy_2_0_2 =          - Qdydt * Wy_2_0_2;
          //   const auto jy_2_1_2 = jy_2_0_2 - Qdydt * Wy_2_1_2;
          //   const auto jy_2_2_2 = jy_2_1_2 - Qdydt * Wy_2_2_2;
          //   const auto jy_3_0_2 =          - Qdydt * Wy_3_0_2;
          //   const auto jy_3_1_2 = jy_3_0_2 - Qdydt * Wy_3_1_2;
          //   const auto jy_3_2_2 = jy_3_1_2 - Qdydt * Wy_3_2_2;
          //
          //   const auto jy_0_0_3 =          - Qdydt * Wy_0_0_3;
          //   const auto jy_0_1_3 = jy_0_0_3 - Qdydt * Wy_0_1_3;
          //   const auto jy_0_2_3 = jy_0_1_3 - Qdydt * Wy_0_2_3;
          //   const auto jy_1_0_3 =          - Qdydt * Wy_1_0_3;
          //   const auto jy_1_1_3 = jy_1_0_3 - Qdydt * Wy_1_1_3;
          //   const auto jy_1_2_3 = jy_1_1_3 - Qdydt * Wy_1_2_3;
          //   const auto jy_2_0_3 =          - Qdydt * Wy_2_0_3;
          //   const auto jy_2_1_3 = jy_2_0_3 - Qdydt * Wy_2_1_3;
          //   const auto jy_2_2_3 = jy_2_1_3 - Qdydt * Wy_2_2_3;
          //   const auto jy_3_0_3 =          - Qdydt * Wy_3_0_3;
          //   const auto jy_3_1_3 = jy_3_0_3 - Qdydt * Wy_3_1_3;
          //   const auto jy_3_2_3 = jy_3_1_3 - Qdydt * Wy_3_2_3;
          //
          //   /*
          //     z - component
          //   */
          //   const auto Wz_0_0_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_0 * S0y_0 + S1x_0 * S1y_0 +
          //                          HALF * (S0x_0 * S1y_0 + S0y_0 * S1x_0));
          //   const auto Wz_0_0_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_0 * S0y_0 + S1x_0 * S1y_0 +
          //                          HALF * (S0x_0 * S1y_0 + S0y_0 * S1x_0));
          //   const auto Wz_0_0_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_0 * S0y_0 + S1x_0 * S1y_0 +
          //                          HALF * (S0x_0 * S1y_0 + S0y_0 * S1x_0));
          //
          //   const auto Wz_0_1_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_0 * S0y_1 + S1x_0 * S1y_1 +
          //                          HALF * (S0x_0 * S1y_1 + S0y_1 * S1x_0));
          //   const auto Wz_0_1_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_0 * S0y_1 + S1x_0 * S1y_1 +
          //                          HALF * (S0x_0 * S1y_1 + S0y_1 * S1x_0));
          //   const auto Wz_0_1_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_0 * S0y_1 + S1x_0 * S1y_1 +
          //                          HALF * (S0x_0 * S1y_1 + S0y_1 * S1x_0));
          //
          //   const auto Wz_0_2_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_0 * S0y_2 + S1x_0 * S1y_2 +
          //                          HALF * (S0x_0 * S1y_2 + S0y_2 * S1x_0));
          //   const auto Wz_0_2_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_0 * S0y_2 + S1x_0 * S1y_2 +
          //                          HALF * (S0x_0 * S1y_2 + S0y_2 * S1x_0));
          //   const auto Wz_0_2_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_0 * S0y_2 + S1x_0 * S1y_2 +
          //                          HALF * (S0x_0 * S1y_2 + S0y_2 * S1x_0));
          //
          //   const auto Wz_0_3_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_0 * S0y_3 + S1x_0 * S1y_3 +
          //                          HALF * (S0x_0 * S1y_3 + S0y_3 * S1x_0));
          //   const auto Wz_0_3_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_0 * S0y_3 + S1x_0 * S1y_3 +
          //                          HALF * (S0x_0 * S1y_3 + S0y_3 * S1x_0));
          //   const auto Wz_0_3_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_0 * S0y_3 + S1x_0 * S1y_3 +
          //                          HALF * (S0x_0 * S1y_3 + S0y_3 * S1x_0));
          //
          //   // Unrolled loop for Wz[i][j][k] with i = 1 and interp_order + 2 = 4
          //   const auto Wz_1_0_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_1 * S0y_0 + S1x_1 * S1y_0 +
          //                          HALF * (S0x_1 * S1y_0 + S0y_0 * S1x_1));
          //   const auto Wz_1_0_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_1 * S0y_0 + S1x_1 * S1y_0 +
          //                          HALF * (S0x_1 * S1y_0 + S0y_0 * S1x_1));
          //   const auto Wz_1_0_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_1 * S0y_0 + S1x_1 * S1y_0 +
          //                          HALF * (S0x_1 * S1y_0 + S0y_0 * S1x_1));
          //
          //   const auto Wz_1_1_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_1 * S0y_1 + S1x_1 * S1y_1 +
          //                          HALF * (S0x_1 * S1y_1 + S0y_1 * S1x_1));
          //   const auto Wz_1_1_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_1 * S0y_1 + S1x_1 * S1y_1 +
          //                          HALF * (S0x_1 * S1y_1 + S0y_1 * S1x_1));
          //   const auto Wz_1_1_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_1 * S0y_1 + S1x_1 * S1y_1 +
          //                          HALF * (S0x_1 * S1y_1 + S0y_1 * S1x_1));
          //
          //   const auto Wz_1_2_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_1 * S0y_2 + S1x_1 * S1y_2 +
          //                          HALF * (S0x_1 * S1y_2 + S0y_2 * S1x_1));
          //   const auto Wz_1_2_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_1 * S0y_2 + S1x_1 * S1y_2 +
          //                          HALF * (S0x_1 * S1y_2 + S0y_2 * S1x_1));
          //   const auto Wz_1_2_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_1 * S0y_2 + S1x_1 * S1y_2 +
          //                          HALF * (S0x_1 * S1y_2 + S0y_2 * S1x_1));
          //
          //   const auto Wz_1_3_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_1 * S0y_3 + S1x_1 * S1y_3 +
          //                          HALF * (S0x_1 * S1y_3 + S0y_3 * S1x_1));
          //   const auto Wz_1_3_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_1 * S0y_3 + S1x_1 * S1y_3 +
          //                          HALF * (S0x_1 * S1y_3 + S0y_3 * S1x_1));
          //   const auto Wz_1_3_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_1 * S0y_3 + S1x_1 * S1y_3 +
          //                          HALF * (S0x_1 * S1y_3 + S0y_3 * S1x_1));
          //
          //   // Unrolled loop for Wz[i][j][k] with i = 2 and interp_order + 2 = 4
          //   const auto Wz_2_0_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_2 * S0y_0 + S1x_2 * S1y_0 +
          //                          HALF * (S0x_2 * S1y_0 + S0y_0 * S1x_2));
          //   const auto Wz_2_0_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_2 * S0y_0 + S1x_2 * S1y_0 +
          //                          HALF * (S0x_2 * S1y_0 + S0y_0 * S1x_2));
          //   const auto Wz_2_0_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_2 * S0y_0 + S1x_2 * S1y_0 +
          //                          HALF * (S0x_2 * S1y_0 + S0y_0 * S1x_2));
          //
          //   const auto Wz_2_1_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_2 * S0y_1 + S1x_2 * S1y_1 +
          //                          HALF * (S0x_2 * S1y_1 + S0y_1 * S1x_2));
          //   const auto Wz_2_1_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_2 * S0y_1 + S1x_2 * S1y_1 +
          //                          HALF * (S0x_2 * S1y_1 + S0y_1 * S1x_2));
          //   const auto Wz_2_1_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_2 * S0y_1 + S1x_2 * S1y_1 +
          //                          HALF * (S0x_2 * S1y_1 + S0y_1 * S1x_2));
          //
          //   const auto Wz_2_2_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_2 * S0y_2 + S1x_2 * S1y_2 +
          //                          HALF * (S0x_2 * S1y_2 + S0y_2 * S1x_2));
          //   const auto Wz_2_2_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_2 * S0y_2 + S1x_2 * S1y_2 +
          //                          HALF * (S0x_2 * S1y_2 + S0y_2 * S1x_2));
          //   const auto Wz_2_2_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_2 * S0y_2 + S1x_2 * S1y_2 +
          //                          HALF * (S0x_2 * S1y_2 + S0y_2 * S1x_2));
          //
          //   const auto Wz_2_3_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_2 * S0y_3 + S1x_2 * S1y_3 +
          //                          HALF * (S0x_2 * S1y_3 + S0y_3 * S1x_2));
          //   const auto Wz_2_3_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_2 * S0y_3 + S1x_2 * S1y_3 +
          //                          HALF * (S0x_2 * S1y_3 + S0y_3 * S1x_2));
          //   const auto Wz_2_3_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_2 * S0y_3 + S1x_2 * S1y_3 +
          //                          HALF * (S0x_2 * S1y_3 + S0y_3 * S1x_2));
          //
          //   // Unrolled loop for Wz[i][j][k] with i = 3 and interp_order + 2 = 4
          //   const auto Wz_3_0_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_3 * S0y_0 + S1x_3 * S1y_0 +
          //                          HALF * (S0x_3 * S1y_0 + S0y_0 * S1x_3));
          //   const auto Wz_3_0_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_3 * S0y_0 + S1x_3 * S1y_0 +
          //                          HALF * (S0x_3 * S1y_0 + S0y_0 * S1x_3));
          //   const auto Wz_3_0_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_3 * S0y_0 + S1x_3 * S1y_0 +
          //                          HALF * (S0x_3 * S1y_0 + S0y_0 * S1x_3));
          //
          //   const auto Wz_3_1_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_3 * S0y_1 + S1x_3 * S1y_1 +
          //                          HALF * (S0x_3 * S1y_1 + S0y_1 * S1x_3));
          //   const auto Wz_3_1_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_3 * S0y_1 + S1x_3 * S1y_1 +
          //                          HALF * (S0x_3 * S1y_1 + S0y_1 * S1x_3));
          //   const auto Wz_3_1_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_3 * S0y_1 + S1x_3 * S1y_1 +
          //                          HALF * (S0x_3 * S1y_1 + S0y_1 * S1x_3));
          //
          //   const auto Wz_3_2_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_3 * S0y_2 + S1x_3 * S1y_2 +
          //                          HALF * (S0x_3 * S1y_2 + S0y_2 * S1x_3));
          //   const auto Wz_3_2_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_3 * S0y_2 + S1x_3 * S1y_2 +
          //                          HALF * (S0x_3 * S1y_2 + S0y_2 * S1x_3));
          //   const auto Wz_3_2_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_3 * S0y_2 + S1x_3 * S1y_2 +
          //                          HALF * (S0x_3 * S1y_2 + S0y_2 * S1x_3));
          //
          //   const auto Wz_3_3_0 = THIRD * (S1z_0 - S0z_0) *
          //                         (S0x_3 * S0y_3 + S1x_3 * S1y_3 +
          //                          HALF * (S0x_3 * S1y_3 + S0y_3 * S1x_3));
          //   const auto Wz_3_3_1 = THIRD * (S1z_1 - S0z_1) *
          //                         (S0x_3 * S0y_3 + S1x_3 * S1y_3 +
          //                          HALF * (S0x_3 * S1y_3 + S0y_3 * S1x_3));
          //   const auto Wz_3_3_2 = THIRD * (S1z_2 - S0z_2) *
          //                         (S0x_3 * S0y_3 + S1x_3 * S1y_3 +
          //                          HALF * (S0x_3 * S1y_3 + S0y_3 * S1x_3));
          //
          //   const real_t Qdzdt = coeff * inv_dt;
          //
          //   const auto jz_0_0_0 =          - Qdzdt * Wz_0_0_0;
          //   const auto jz_0_0_1 = jz_0_0_0 - Qdzdt * Wz_0_0_1;
          //   const auto jz_0_0_2 = jz_0_0_1 - Qdzdt * Wz_0_0_2;
          //   const auto jz_0_1_0 =          - Qdzdt * Wz_0_1_0;
          //   const auto jz_0_1_1 = jz_0_1_0 - Qdzdt * Wz_0_1_1;
          //   const auto jz_0_1_2 = jz_0_1_1 - Qdzdt * Wz_0_1_2;
          //   const auto jz_0_2_0 =          - Qdzdt * Wz_0_2_0;
          //   const auto jz_0_2_1 = jz_0_2_0 - Qdzdt * Wz_0_2_1;
          //   const auto jz_0_2_2 = jz_0_2_1 - Qdzdt * Wz_0_2_2;
          //   const auto jz_0_3_0 =          - Qdzdt * Wz_0_3_0;
          //   const auto jz_0_3_1 = jz_0_3_0 - Qdzdt * Wz_0_3_1;
          //   const auto jz_0_3_2 = jz_0_3_1 - Qdzdt * Wz_0_3_2;
          //
          //   const auto jz_1_0_0 =          - Qdzdt * Wz_1_0_0;
          //   const auto jz_1_0_1 = jz_1_0_0 - Qdzdt * Wz_1_0_1;
          //   const auto jz_1_0_2 = jz_1_0_1 - Qdzdt * Wz_1_0_2;
          //   const auto jz_1_1_0 =          - Qdzdt * Wz_1_1_0;
          //   const auto jz_1_1_1 = jz_1_1_0 - Qdzdt * Wz_1_1_1;
          //   const auto jz_1_1_2 = jz_1_1_1 - Qdzdt * Wz_1_1_2;
          //   const auto jz_1_2_0 =          - Qdzdt * Wz_1_2_0;
          //   const auto jz_1_2_1 = jz_1_2_0 - Qdzdt * Wz_1_2_1;
          //   const auto jz_1_2_2 = jz_1_2_1 - Qdzdt * Wz_1_2_2;
          //   const auto jz_1_3_0 =          - Qdzdt * Wz_1_3_0;
          //   const auto jz_1_3_1 = jz_1_3_0 - Qdzdt * Wz_1_3_1;
          //   const auto jz_1_3_2 = jz_1_3_1 - Qdzdt * Wz_1_3_2;
          //
          //   const auto jz_2_0_0 =          - Qdzdt * Wz_2_0_0;
          //   const auto jz_2_0_1 = jz_2_0_0 - Qdzdt * Wz_2_0_1;
          //   const auto jz_2_0_2 = jz_2_0_1 - Qdzdt * Wz_2_0_2;
          //   const auto jz_2_1_0 =          - Qdzdt * Wz_2_1_0;
          //   const auto jz_2_1_1 = jz_2_1_0 - Qdzdt * Wz_2_1_1;
          //   const auto jz_2_1_2 = jz_2_1_1 - Qdzdt * Wz_2_1_2;
          //   const auto jz_2_2_0 =          - Qdzdt * Wz_2_2_0;
          //   const auto jz_2_2_1 = jz_2_2_0 - Qdzdt * Wz_2_2_1;
          //   const auto jz_2_2_2 = jz_2_2_1 - Qdzdt * Wz_2_2_2;
          //   const auto jz_2_3_0 =          - Qdzdt * Wz_2_3_0;
          //   const auto jz_2_3_1 = jz_2_3_0 - Qdzdt * Wz_2_3_1;
          //   const auto jz_2_3_2 = jz_2_3_1 - Qdzdt * Wz_2_3_2;
          //
          //   const auto jz_3_0_0 =          - Qdzdt * Wz_3_0_0;
          //   const auto jz_3_0_1 = jz_3_0_0 - Qdzdt * Wz_3_0_1;
          //   const auto jz_3_0_2 = jz_3_0_1 - Qdzdt * Wz_3_0_2;
          //   const auto jz_3_1_0 =          - Qdzdt * Wz_3_1_0;
          //   const auto jz_3_1_1 = jz_3_1_0 - Qdzdt * Wz_3_1_1;
          //   const auto jz_3_1_2 = jz_3_1_1 - Qdzdt * Wz_3_1_2;
          //   const auto jz_3_2_0 =          - Qdzdt * Wz_3_2_0;
          //   const auto jz_3_2_1 = jz_3_2_0 - Qdzdt * Wz_3_2_1;
          //   const auto jz_3_2_2 = jz_3_2_1 - Qdzdt * Wz_3_2_2;
          //   const auto jz_3_3_0 =          - Qdzdt * Wz_3_3_0;
          //   const auto jz_3_3_1 = jz_3_3_0 - Qdzdt * Wz_3_3_1;
          //   const auto jz_3_3_2 = jz_3_3_1 - Qdzdt * Wz_3_3_2;
          //
          //
          //   /*
          //     Current update
          //   */
          //   auto J_acc = J.access();
          //
          //   J_acc(ix_min,     iy_min,     iz_min,     cur::jx1) += jx_0_0_0;
          //   J_acc(ix_min,     iy_min,     iz_min + 1, cur::jx1) += jx_0_0_1;
          //   J_acc(ix_min,     iy_min,     iz_min + 2, cur::jx1) += jx_0_0_2;
          //   J_acc(ix_min,     iy_min + 1, iz_min,     cur::jx1) += jx_0_1_0;
          //   J_acc(ix_min,     iy_min + 1, iz_min + 1, cur::jx1) += jx_0_1_1;
          //   J_acc(ix_min,     iy_min + 1, iz_min + 2, cur::jx1) += jx_0_1_2;
          //   J_acc(ix_min,     iy_min + 2, iz_min,     cur::jx1) += jx_0_2_0;
          //   J_acc(ix_min,     iy_min + 2, iz_min + 1, cur::jx1) += jx_0_2_1;
          //   J_acc(ix_min,     iy_min + 2, iz_min + 2, cur::jx1) += jx_0_2_2;
          //   J_acc(ix_min + 1, iy_min,     iz_min,     cur::jx1) += jx_1_0_0;
          //   J_acc(ix_min + 1, iy_min,     iz_min + 1, cur::jx1) += jx_1_0_1;
          //   J_acc(ix_min + 1, iy_min,     iz_min + 2, cur::jx1) += jx_1_0_2;
          //   J_acc(ix_min + 1, iy_min + 1, iz_min,     cur::jx1) += jx_1_1_0;
          //   J_acc(ix_min + 1, iy_min + 1, iz_min + 1, cur::jx1) += jx_1_1_1;
          //   J_acc(ix_min + 1, iy_min + 1, iz_min + 2, cur::jx1) += jx_1_1_2;
          //   J_acc(ix_min + 1, iy_min + 2, iz_min,     cur::jx1) += jx_1_2_0;
          //   J_acc(ix_min + 1, iy_min + 2, iz_min + 1, cur::jx1) += jx_1_2_1;
          //   J_acc(ix_min + 1, iy_min + 2, iz_min + 2, cur::jx1) += jx_1_2_2;
          //
          //   if (update_x2)
          //   {
          //     J_acc(ix_min + 2, iy_min,     iz_min,     cur::jx1) += jx_2_0_0;
          //     J_acc(ix_min + 2, iy_min,     iz_min + 1, cur::jx1) += jx_2_0_1;
          //     J_acc(ix_min + 2, iy_min,     iz_min + 2, cur::jx1) += jx_2_0_2;
          //     J_acc(ix_min + 2, iy_min + 1, iz_min,     cur::jx1) += jx_2_1_0;
          //     J_acc(ix_min + 2, iy_min + 1, iz_min + 1, cur::jx1) += jx_2_1_1;
          //     J_acc(ix_min + 2, iy_min + 1, iz_min + 2, cur::jx1) += jx_2_1_2;
          //     J_acc(ix_min + 2, iy_min + 2, iz_min,     cur::jx1) += jx_2_2_0;
          //     J_acc(ix_min + 2, iy_min + 2, iz_min + 1, cur::jx1) += jx_2_2_1;
          //     J_acc(ix_min + 2, iy_min + 2, iz_min + 2, cur::jx1) += jx_2_2_2;
          //
          //     if (update_y2)
          //     {
          //       J_acc(ix_min + 2, iy_min + 3, iz_min,     cur::jx1) += jx_2_3_0;
          //       J_acc(ix_min + 2, iy_min + 3, iz_min + 1, cur::jx1) += jx_2_3_1;
          //       J_acc(ix_min + 2, iy_min + 3, iz_min + 2, cur::jx1) += jx_2_3_2;
          //     }
          //
          //     if (update_z2)
          //     {
          //       J_acc(ix_min + 2, iy_min,     iz_min + 3, cur::jx1) += jx_2_0_3;
          //       J_acc(ix_min + 2, iy_min + 1, iz_min + 3, cur::jx1) += jx_2_1_3;
          //       J_acc(ix_min + 2, iy_min + 2, iz_min + 3, cur::jx1) += jx_2_2_3;
          //
          //       if (update_y2)
          //       {
          //         J_acc(ix_min + 2, iy_min + 3, iz_min + 3, cur::jx1) += jx_2_3_3;
          //       }
          //     }
          //   }
          //   //
          //   if (update_y2)
          //   {
          //     J_acc(ix_min,     iy_min + 3, iz_min,     cur::jx1) += jx_0_3_0;
          //     J_acc(ix_min,     iy_min + 3, iz_min + 1, cur::jx1) += jx_0_3_1;
          //     J_acc(ix_min,     iy_min + 3, iz_min + 2, cur::jx1) += jx_0_3_2;
          //     J_acc(ix_min + 1, iy_min + 3, iz_min,     cur::jx1) += jx_1_3_0;
          //     J_acc(ix_min + 1, iy_min + 3, iz_min + 1, cur::jx1) += jx_1_3_1;
          //     J_acc(ix_min + 1, iy_min + 3, iz_min + 2, cur::jx1) += jx_1_3_2;
          //   }
          //
          //   if (update_z2)
          //   {
          //     J_acc(ix_min,     iy_min,     iz_min + 3, cur::jx1) += jx_0_0_3;
          //     J_acc(ix_min,     iy_min + 1, iz_min + 3, cur::jx1) += jx_0_1_3;
          //     J_acc(ix_min,     iy_min + 2, iz_min + 3, cur::jx1) += jx_0_2_3;
          //     J_acc(ix_min + 1, iy_min,     iz_min + 3, cur::jx1) += jx_1_0_3;
          //     J_acc(ix_min + 1, iy_min + 1, iz_min + 3, cur::jx1) += jx_1_1_3;
          //     J_acc(ix_min + 1, iy_min + 2, iz_min + 3, cur::jx1) += jx_1_2_3;
          //
          //     if (update_y2)
          //     {
          //       J_acc(ix_min,     iy_min + 3, iz_min + 3, cur::jx1) += jx_0_3_3;
          //       J_acc(ix_min + 1, iy_min + 3, iz_min + 3, cur::jx1) += jx_1_3_3;
          //     }
          //   }
          //
          //
          //   /*
          //     y-component
          //   */
          //   J_acc(ix_min,     iy_min,     iz_min,     cur::jx2) += jy_0_0_0;
          //   J_acc(ix_min,     iy_min,     iz_min + 1, cur::jx2) += jy_0_0_1;
          //   J_acc(ix_min,     iy_min,     iz_min + 2, cur::jx2) += jy_0_0_2;
          //   J_acc(ix_min,     iy_min + 1, iz_min,     cur::jx2) += jy_0_1_0;
          //   J_acc(ix_min,     iy_min + 1, iz_min + 1, cur::jx2) += jy_0_1_1;
          //   J_acc(ix_min,     iy_min + 1, iz_min + 2, cur::jx2) += jy_0_1_2;
          //   J_acc(ix_min + 1, iy_min,     iz_min,     cur::jx2) += jy_1_0_0;
          //   J_acc(ix_min + 1, iy_min,     iz_min + 1, cur::jx2) += jy_1_0_1;
          //   J_acc(ix_min + 1, iy_min,     iz_min + 2, cur::jx2) += jy_1_0_2;
          //   J_acc(ix_min + 1, iy_min + 1, iz_min,     cur::jx2) += jy_1_1_0;
          //   J_acc(ix_min + 1, iy_min + 1, iz_min + 1, cur::jx2) += jy_1_1_1;
          //   J_acc(ix_min + 1, iy_min + 1, iz_min + 2, cur::jx2) += jy_1_1_2;
          //   J_acc(ix_min + 2, iy_min,     iz_min,     cur::jx2) += jy_2_0_0;
          //   J_acc(ix_min + 2, iy_min,     iz_min + 1, cur::jx2) += jy_2_0_1;
          //   J_acc(ix_min + 2, iy_min,     iz_min + 2, cur::jx2) += jy_2_0_2;
          //   J_acc(ix_min + 2, iy_min + 1, iz_min,     cur::jx2) += jy_2_1_0;
          //   J_acc(ix_min + 2, iy_min + 1, iz_min + 1, cur::jx2) += jy_2_1_1;
          //   J_acc(ix_min + 2, iy_min + 1, iz_min + 2, cur::jx2) += jy_2_1_2;
          //
          //   if (update_x2)
          //   {
          //     J_acc(ix_min + 3, iy_min,     iz_min,     cur::jx2) += jy_3_0_0;
          //     J_acc(ix_min + 3, iy_min,     iz_min + 1, cur::jx2) += jy_3_0_1;
          //     J_acc(ix_min + 3, iy_min,     iz_min + 2, cur::jx2) += jy_3_0_2;
          //     J_acc(ix_min + 3, iy_min + 1, iz_min,     cur::jx2) += jy_3_1_0;
          //     J_acc(ix_min + 3, iy_min + 1, iz_min + 1, cur::jx2) += jy_3_1_1;
          //     J_acc(ix_min + 3, iy_min + 1, iz_min + 2, cur::jx2) += jy_3_1_2;
          //
          //     if (update_z2)
          //     {
          //       J_acc(ix_min + 3, iy_min,     iz_min + 3, cur::jx2) += jy_3_0_3;
          //       J_acc(ix_min + 3, iy_min + 1, iz_min + 3, cur::jx2) += jy_3_1_3;
          //     }
          //   }
          //
          //   if (update_y2)
          //   {
          //     J_acc(ix_min,     iy_min + 2, iz_min,     cur::jx2) += jy_0_2_0;
          //     J_acc(ix_min,     iy_min + 2, iz_min + 1, cur::jx2) += jy_0_2_1;
          //     J_acc(ix_min,     iy_min + 2, iz_min + 2, cur::jx2) += jy_0_2_2;
          //     J_acc(ix_min + 1, iy_min + 2, iz_min,     cur::jx2) += jy_1_2_0;
          //     J_acc(ix_min + 1, iy_min + 2, iz_min + 1, cur::jx2) += jy_1_2_1;
          //     J_acc(ix_min + 1, iy_min + 2, iz_min + 2, cur::jx2) += jy_1_2_2;
          //     J_acc(ix_min + 2, iy_min + 2, iz_min,     cur::jx2) += jy_2_2_0;
          //     J_acc(ix_min + 2, iy_min + 2, iz_min + 1, cur::jx2) += jy_2_2_1;
          //     J_acc(ix_min + 2, iy_min + 2, iz_min + 2, cur::jx2) += jy_2_2_2;
          //
          //     if (update_x2)
          //     {
          //       J_acc(ix_min + 3, iy_min + 2, iz_min,     cur::jx2) += jy_3_2_0;
          //       J_acc(ix_min + 3, iy_min + 2, iz_min + 1, cur::jx2) += jy_3_2_1;
          //       J_acc(ix_min + 3, iy_min + 2, iz_min + 2, cur::jx2) += jy_3_2_2;
          //
          //       if (update_z2)
          //       {
          //         J_acc(ix_min + 2, iy_min + 2, iz_min + 3, cur::jx2) += jy_2_2_3;
          //         J_acc(ix_min + 3, iy_min + 2, iz_min + 3, cur::jx2) += jy_3_2_3;
          //       }
          //     }
          //
          //     if (update_z2)
          //     {
          //       J_acc(ix_min,     iy_min + 2, iz_min + 3, cur::jx2) += jy_0_2_3;
          //       J_acc(ix_min + 1, iy_min + 2, iz_min + 3, cur::jx2) += jy_1_2_3;
          //     }
          //   }
          //
          //   if (update_z2)
          //   {
          //     J_acc(ix_min,     iy_min,     iz_min + 3, cur::jx2) += jy_0_0_3;
          //     J_acc(ix_min,     iy_min + 1, iz_min + 3, cur::jx2) += jy_0_1_3;
          //     J_acc(ix_min + 1, iy_min,     iz_min + 3, cur::jx2) += jy_1_0_3;
          //     J_acc(ix_min + 1, iy_min + 1, iz_min + 3, cur::jx2) += jy_1_1_3;
          //     J_acc(ix_min + 2, iy_min,     iz_min + 3, cur::jx2) += jy_2_0_3;
          //     J_acc(ix_min + 2, iy_min + 1, iz_min + 3, cur::jx2) += jy_2_1_3;
          //   }
          //
          //   /*
          //     z-component
          //   */
          //   J_acc(ix_min,     iy_min,     iz_min,     cur::jx3) += jz_0_0_0;
          //   J_acc(ix_min,     iy_min,     iz_min + 1, cur::jx3) += jz_0_0_1;
          //   J_acc(ix_min,     iy_min + 1, iz_min,     cur::jx3) += jz_0_1_0;
          //   J_acc(ix_min,     iy_min + 1, iz_min + 1, cur::jx3) += jz_0_1_1;
          //   J_acc(ix_min,     iy_min + 2, iz_min,     cur::jx3) += jz_0_2_0;
          //   J_acc(ix_min,     iy_min + 2, iz_min + 1, cur::jx3) += jz_0_2_1;
          //   J_acc(ix_min + 1, iy_min,     iz_min,     cur::jx3) += jz_1_0_0;
          //   J_acc(ix_min + 1, iy_min,     iz_min + 1, cur::jx3) += jz_1_0_1;
          //   J_acc(ix_min + 1, iy_min + 1, iz_min,     cur::jx3) += jz_1_1_0;
          //   J_acc(ix_min + 1, iy_min + 1, iz_min + 1, cur::jx3) += jz_1_1_1;
          //   J_acc(ix_min + 1, iy_min + 2, iz_min,     cur::jx3) += jz_1_2_0;
          //   J_acc(ix_min + 1, iy_min + 2, iz_min + 1, cur::jx3) += jz_1_2_1;
          //   J_acc(ix_min + 2, iy_min,     iz_min,     cur::jx3) += jz_2_0_0;
          //   J_acc(ix_min + 2, iy_min,     iz_min + 1, cur::jx3) += jz_2_0_1;
          //   J_acc(ix_min + 2, iy_min + 1, iz_min,     cur::jx3) += jz_2_1_0;
          //   J_acc(ix_min + 2, iy_min + 1, iz_min + 1, cur::jx3) += jz_2_1_1;
          //   J_acc(ix_min + 2, iy_min + 2, iz_min,     cur::jx3) += jz_2_2_0;
          //   J_acc(ix_min + 2, iy_min + 2, iz_min + 1, cur::jx3) += jz_2_2_1;
          //
          //   if (update_x2)
          //   {
          //     J_acc(ix_min + 3, iy_min,     iz_min,     cur::jx3) += jz_3_0_0;
          //     J_acc(ix_min + 3, iy_min,     iz_min + 1, cur::jx3) += jz_3_0_1;
          //     J_acc(ix_min + 3, iy_min + 1, iz_min,     cur::jx3) += jz_3_1_0;
          //     J_acc(ix_min + 3, iy_min + 1, iz_min + 1, cur::jx3) += jz_3_1_1;
          //     J_acc(ix_min + 3, iy_min + 2, iz_min,     cur::jx3) += jz_3_2_0;
          //     J_acc(ix_min + 3, iy_min + 2, iz_min + 1, cur::jx3) += jz_3_2_1;
          //     J_acc(ix_min + 3, iy_min + 3, iz_min,     cur::jx3) += jz_3_3_0;
          //     J_acc(ix_min + 3, iy_min + 3, iz_min + 1, cur::jx3) += jz_3_3_1;
          //   }
          //
          //   if (update_y2)
          //   {
          //     J_acc(ix_min,     iy_min + 3, iz_min,     cur::jx3) += jz_0_3_0;
          //     J_acc(ix_min,     iy_min + 3, iz_min + 1, cur::jx3) += jz_0_3_1;
          //     J_acc(ix_min + 1, iy_min + 3, iz_min,     cur::jx3) += jz_1_3_0;
          //     J_acc(ix_min + 1, iy_min + 3, iz_min + 1, cur::jx3) += jz_1_3_1;
          //     J_acc(ix_min + 2, iy_min + 3, iz_min,     cur::jx3) += jz_2_3_0;
          //     J_acc(ix_min + 2, iy_min + 3, iz_min + 1, cur::jx3) += jz_2_3_1;
          //   }
          //
          //   if (update_z2)
          //   {
          //     J_acc(ix_min,     iy_min,     iz_min + 2, cur::jx3) += jz_0_0_2;
          //     J_acc(ix_min,     iy_min + 1, iz_min + 2, cur::jx3) += jz_0_1_2;
          //     J_acc(ix_min,     iy_min + 2, iz_min + 2, cur::jx3) += jz_0_2_2;
          //     J_acc(ix_min + 1, iy_min,     iz_min + 2, cur::jx3) += jz_1_0_2;
          //     J_acc(ix_min + 1, iy_min + 1, iz_min + 2, cur::jx3) += jz_1_1_2;
          //     J_acc(ix_min + 1, iy_min + 2, iz_min + 2, cur::jx3) += jz_1_2_2;
          //     J_acc(ix_min + 2, iy_min,     iz_min + 2, cur::jx3) += jz_2_0_2;
          //     J_acc(ix_min + 2, iy_min + 1, iz_min + 2, cur::jx3) += jz_2_1_2;
          //     J_acc(ix_min + 2, iy_min + 2, iz_min + 2, cur::jx3) += jz_2_2_2;
          //
          //     if (update_x2)
          //     {
          //       J_acc(ix_min + 3, iy_min,     iz_min + 2, cur::jx3) += jz_3_0_2;
          //       J_acc(ix_min + 3, iy_min + 1, iz_min + 2, cur::jx3) += jz_3_1_2;
          //       J_acc(ix_min + 3, iy_min + 2, iz_min + 2, cur::jx3) += jz_3_2_2;
          //
          //       if (update_y2)
          //       {
          //         J_acc(ix_min + 3, iy_min + 3, iz_min + 2, cur::jx3) += jz_3_3_2;
          //       }
          //     }
          //
          //     if (update_y2)
          //     {
          //       J_acc(ix_min,     iy_min + 3, iz_min + 2, cur::jx3) += jz_0_3_2;
          //       J_acc(ix_min + 1, iy_min + 3, iz_min + 2, cur::jx3) += jz_1_3_2;
          //       J_acc(ix_min + 2, iy_min + 3, iz_min + 2, cur::jx3) += jz_2_3_2;
          //     }
          //   }
          // clang-format on
        } // dimension

      } else if constexpr (O == 3u) {
        /*
          Higher order charge conserving current deposition based on
          Esirkepov (2001) https://ui.adsabs.harvard.edu/abs/2001CoPhC.135..144E/abstract

          We need to define the follwowing variable:
          - Shape functions in spatial directions for the particle position
            before and after the current timestep.
            S0_*, S1_*
          - Density composition matrix
            Wx_*, Wy_*, Wz_*
        */

        /*
            x - direction
        */

        // shape function at previous timestep
        real_t   S0x_0, S0x_1, S0x_2, S0x_3, S0x_4;
        // shape function at current timestep
        real_t   S1x_0, S1x_1, S1x_2, S1x_3, S1x_4;
        // indices of the shape function
        ncells_t ix_min;
        bool     update_x3;
        // find indices and define shape function
        // clang-format off
        shape_function_3rd(S0x_0, S0x_1, S0x_2, S0x_3, S0x_4,
                           S1x_0, S1x_1, S1x_2, S1x_3, S1x_4,
                           ix_min, update_x3,
                           i1(p), dx1(p),
                           i1_prev(p), dx1_prev(p));
        // clang-format on

        if constexpr (D == Dim::_1D) {
          // ToDo
        } else if constexpr (D == Dim::_2D) {

          /*
            y - direction
          */

          // shape function at previous timestep
          real_t   S0y_0, S0y_1, S0y_2, S0y_3, S0y_4;
          // shape function at current timestep
          real_t   S1y_0, S1y_1, S1y_2, S1y_3, S1y_4;
          // indices of the shape function
          ncells_t iy_min;
          bool     update_y3;
          // find indices and define shape function
          // clang-format off
          shape_function_3rd(S0y_0, S0y_1, S0y_2, S0y_3, S0y_4,
                             S1y_0, S1y_1, S1y_2, S1y_3, S1y_4,
                             iy_min, update_y3,
                             i2(p), dx2(p),
                             i2_prev(p), dx2_prev(p));
          // clang-format on

          // Esirkepov 2001, Eq. 38
          /*
              x - component
          */
          // Calculate weight function - unrolled
          const auto Wx_0_0 = HALF * (S1x_0 - S0x_0) * (S0y_0 + S1y_0);
          const auto Wx_0_1 = HALF * (S1x_0 - S0x_0) * (S0y_1 + S1y_1);
          const auto Wx_0_2 = HALF * (S1x_0 - S0x_0) * (S0y_2 + S1y_2);
          const auto Wx_0_3 = HALF * (S1x_0 - S0x_0) * (S0y_3 + S1y_3);
          const auto Wx_0_4 = HALF * (S1x_0 - S0x_0) * (S0y_4 + S1y_4);

          const auto Wx_1_0 = HALF * (S1x_1 - S0x_1) * (S0y_0 + S1y_0);
          const auto Wx_1_1 = HALF * (S1x_1 - S0x_1) * (S0y_1 + S1y_1);
          const auto Wx_1_2 = HALF * (S1x_1 - S0x_1) * (S0y_2 + S1y_2);
          const auto Wx_1_3 = HALF * (S1x_1 - S0x_1) * (S0y_3 + S1y_3);
          const auto Wx_1_4 = HALF * (S1x_1 - S0x_1) * (S0y_4 + S1y_4);

          const auto Wx_2_0 = HALF * (S1x_2 - S0x_2) * (S0y_0 + S1y_0);
          const auto Wx_2_1 = HALF * (S1x_2 - S0x_2) * (S0y_1 + S1y_1);
          const auto Wx_2_2 = HALF * (S1x_2 - S0x_2) * (S0y_2 + S1y_2);
          const auto Wx_2_3 = HALF * (S1x_2 - S0x_2) * (S0y_3 + S1y_3);
          const auto Wx_2_4 = HALF * (S1x_2 - S0x_2) * (S0y_4 + S1y_4);

          const auto Wx_3_0 = HALF * (S1x_3 - S0x_3) * (S0y_0 + S1y_0);
          const auto Wx_3_1 = HALF * (S1x_3 - S0x_3) * (S0y_1 + S1y_1);
          const auto Wx_3_2 = HALF * (S1x_3 - S0x_3) * (S0y_2 + S1y_2);
          const auto Wx_3_3 = HALF * (S1x_3 - S0x_3) * (S0y_3 + S1y_3);
          const auto Wx_3_4 = HALF * (S1x_3 - S0x_3) * (S0y_4 + S1y_4);

          // Unrolled calculations for Wy
          const auto Wy_0_0 = HALF * (S1x_0 + S0x_0) * (S1y_0 - S0y_0);
          const auto Wy_0_1 = HALF * (S1x_0 + S0x_0) * (S1y_1 - S0y_1);
          const auto Wy_0_2 = HALF * (S1x_0 + S0x_0) * (S1y_2 - S0y_2);
          const auto Wy_0_3 = HALF * (S1x_0 + S0x_0) * (S1y_3 - S0y_3);

          const auto Wy_1_0 = HALF * (S1x_1 + S0x_1) * (S1y_0 - S0y_0);
          const auto Wy_1_1 = HALF * (S1x_1 + S0x_1) * (S1y_1 - S0y_1);
          const auto Wy_1_2 = HALF * (S1x_1 + S0x_1) * (S1y_2 - S0y_2);
          const auto Wy_1_3 = HALF * (S1x_1 + S0x_1) * (S1y_3 - S0y_3);

          const auto Wy_2_0 = HALF * (S1x_2 + S0x_2) * (S1y_0 - S0y_0);
          const auto Wy_2_1 = HALF * (S1x_2 + S0x_2) * (S1y_1 - S0y_1);
          const auto Wy_2_2 = HALF * (S1x_2 + S0x_2) * (S1y_2 - S0y_2);
          const auto Wy_2_3 = HALF * (S1x_2 + S0x_2) * (S1y_3 - S0y_3);

          const auto Wy_3_0 = HALF * (S1x_3 + S0x_3) * (S1y_0 - S0y_0);
          const auto Wy_3_1 = HALF * (S1x_3 + S0x_3) * (S1y_1 - S0y_1);
          const auto Wy_3_2 = HALF * (S1x_3 + S0x_3) * (S1y_2 - S0y_2);
          const auto Wy_3_3 = HALF * (S1x_3 + S0x_3) * (S1y_3 - S0y_3);

          const auto Wy_4_0 = HALF * (S1x_4 + S0x_4) * (S1y_0 - S0y_0);
          const auto Wy_4_1 = HALF * (S1x_4 + S0x_4) * (S1y_1 - S0y_1);
          const auto Wy_4_2 = HALF * (S1x_4 + S0x_4) * (S1y_2 - S0y_2);
          const auto Wy_4_3 = HALF * (S1x_4 + S0x_4) * (S1y_3 - S0y_3);

          // Unrolled calculations for Wz
          const auto Wz_0_0 = THIRD * (S1y_0 * (HALF * S0x_0 + S1x_0) +
                                       S0y_0 * (HALF * S1x_0 + S0x_0));
          const auto Wz_0_1 = THIRD * (S1y_1 * (HALF * S0x_0 + S1x_0) +
                                       S0y_1 * (HALF * S1x_0 + S0x_0));
          const auto Wz_0_2 = THIRD * (S1y_2 * (HALF * S0x_0 + S1x_0) +
                                       S0y_2 * (HALF * S1x_0 + S0x_0));
          const auto Wz_0_3 = THIRD * (S1y_3 * (HALF * S0x_0 + S1x_0) +
                                       S0y_3 * (HALF * S1x_0 + S0x_0));
          const auto Wz_0_4 = THIRD * (S1y_4 * (HALF * S0x_0 + S1x_0) +
                                       S0y_4 * (HALF * S1x_0 + S0x_0));

          const auto Wz_1_0 = THIRD * (S1y_0 * (HALF * S0x_1 + S1x_1) +
                                       S0y_0 * (HALF * S1x_1 + S0x_1));
          const auto Wz_1_1 = THIRD * (S1y_1 * (HALF * S0x_1 + S1x_1) +
                                       S0y_1 * (HALF * S1x_1 + S0x_1));
          const auto Wz_1_2 = THIRD * (S1y_2 * (HALF * S0x_1 + S1x_1) +
                                       S0y_2 * (HALF * S1x_1 + S0x_1));
          const auto Wz_1_3 = THIRD * (S1y_3 * (HALF * S0x_1 + S1x_1) +
                                       S0y_3 * (HALF * S1x_1 + S0x_1));
          const auto Wz_1_4 = THIRD * (S1y_4 * (HALF * S0x_1 + S1x_1) +
                                       S0y_4 * (HALF * S1x_1 + S0x_1));

          const auto Wz_2_0 = THIRD * (S1y_0 * (HALF * S0x_2 + S1x_2) +
                                       S0y_0 * (HALF * S1x_2 + S0x_2));
          const auto Wz_2_1 = THIRD * (S1y_1 * (HALF * S0x_2 + S1x_2) +
                                       S0y_1 * (HALF * S1x_2 + S0x_2));
          const auto Wz_2_2 = THIRD * (S1y_2 * (HALF * S0x_2 + S1x_2) +
                                       S0y_2 * (HALF * S1x_2 + S0x_2));
          const auto Wz_2_3 = THIRD * (S1y_3 * (HALF * S0x_2 + S1x_2) +
                                       S0y_3 * (HALF * S1x_2 + S0x_2));
          const auto Wz_2_4 = THIRD * (S1y_4 * (HALF * S0x_2 + S1x_2) +
                                       S0y_4 * (HALF * S1x_2 + S0x_2));

          const auto Wz_3_0 = THIRD * (S1y_0 * (HALF * S0x_3 + S1x_3) +
                                       S0y_0 * (HALF * S1x_3 + S0x_3));
          const auto Wz_3_1 = THIRD * (S1y_1 * (HALF * S0x_3 + S1x_3) +
                                       S0y_1 * (HALF * S1x_3 + S0x_3));
          const auto Wz_3_2 = THIRD * (S1y_2 * (HALF * S0x_3 + S1x_3) +
                                       S0y_2 * (HALF * S1x_3 + S0x_3));
          const auto Wz_3_3 = THIRD * (S1y_3 * (HALF * S0x_3 + S1x_3) +
                                       S0y_3 * (HALF * S1x_3 + S0x_3));
          const auto Wz_3_4 = THIRD * (S1y_4 * (HALF * S0x_3 + S1x_3) +
                                       S0y_4 * (HALF * S1x_3 + S0x_3));

          const auto Wz_4_0 = THIRD * (S1y_0 * (HALF * S0x_4 + S1x_4) +
                                       S0y_0 * (HALF * S1x_4 + S0x_4));
          const auto Wz_4_1 = THIRD * (S1y_1 * (HALF * S0x_4 + S1x_4) +
                                       S0y_1 * (HALF * S1x_4 + S0x_4));
          const auto Wz_4_2 = THIRD * (S1y_2 * (HALF * S0x_4 + S1x_4) +
                                       S0y_2 * (HALF * S1x_4 + S0x_4));
          const auto Wz_4_3 = THIRD * (S1y_3 * (HALF * S0x_4 + S1x_4) +
                                       S0y_3 * (HALF * S1x_4 + S0x_4));
          const auto Wz_4_4 = THIRD * (S1y_4 * (HALF * S0x_4 + S1x_4) +
                                       S0y_4 * (HALF * S1x_4 + S0x_4));

          const real_t Qdxdt = coeff * inv_dt;
          const real_t Qdydt = coeff * inv_dt;
          const real_t QVz   = coeff * inv_dt * vp[2];

          // Esirkepov - Eq. 39
          // x-component
          const auto jx_0_0 = -Qdxdt * Wx_0_0;
          const auto jx_1_0 = jx_0_0 - Qdxdt * Wx_1_0;
          const auto jx_2_0 = jx_1_0 - Qdxdt * Wx_2_0;
          const auto jx_3_0 = jx_2_0 - Qdxdt * Wx_3_0;

          const auto jx_0_1 = -Qdxdt * Wx_0_1;
          const auto jx_1_1 = jx_0_1 - Qdxdt * Wx_1_1;
          const auto jx_2_1 = jx_1_1 - Qdxdt * Wx_2_1;
          const auto jx_3_1 = jx_2_1 - Qdxdt * Wx_3_1;

          const auto jx_0_2 = -Qdxdt * Wx_0_2;
          const auto jx_1_2 = jx_0_2 - Qdxdt * Wx_1_2;
          const auto jx_2_2 = jx_1_2 - Qdxdt * Wx_2_2;
          const auto jx_3_2 = jx_2_2 - Qdxdt * Wx_3_2;

          const auto jx_0_3 = -Qdxdt * Wx_0_3;
          const auto jx_1_3 = jx_0_3 - Qdxdt * Wx_1_3;
          const auto jx_2_3 = jx_1_3 - Qdxdt * Wx_2_3;
          const auto jx_3_3 = jx_2_3 - Qdxdt * Wx_3_3;

          const auto jx_0_4 = -Qdxdt * Wx_0_4;
          const auto jx_1_4 = jx_0_4 - Qdxdt * Wx_1_4;
          const auto jx_2_4 = jx_1_4 - Qdxdt * Wx_2_4;
          const auto jx_3_4 = jx_2_4 - Qdxdt * Wx_3_4;

          // y-component
          const auto jy_0_0 = -Qdydt * Wy_0_0;
          const auto jy_0_1 = jy_0_0 - Qdydt * Wy_0_1;
          const auto jy_0_2 = jy_0_1 - Qdydt * Wy_0_2;
          const auto jy_0_3 = jy_0_2 - Qdydt * Wy_0_3;

          const auto jy_1_0 = -Qdydt * Wy_1_0;
          const auto jy_1_1 = jy_1_0 - Qdydt * Wy_1_1;
          const auto jy_1_2 = jy_1_1 - Qdydt * Wy_1_2;
          const auto jy_1_3 = jy_1_2 - Qdydt * Wy_1_3;

          const auto jy_2_0 = -Qdydt * Wy_2_0;
          const auto jy_2_1 = jy_2_0 - Qdydt * Wy_2_1;
          const auto jy_2_2 = jy_2_1 - Qdydt * Wy_2_2;
          const auto jy_2_3 = jy_2_2 - Qdydt * Wy_2_3;

          const auto jy_3_0 = -Qdydt * Wy_3_0;
          const auto jy_3_1 = jy_3_0 - Qdydt * Wy_3_1;
          const auto jy_3_2 = jy_3_1 - Qdydt * Wy_3_2;
          const auto jy_3_3 = jy_3_2 - Qdydt * Wy_3_3;

          const auto jy_4_0 = -Qdydt * Wy_4_0;
          const auto jy_4_1 = jy_4_0 - Qdydt * Wy_4_1;
          const auto jy_4_2 = jy_4_1 - Qdydt * Wy_4_2;
          const auto jy_4_3 = jy_4_2 - Qdydt * Wy_4_3;

          /*
            Current update
          */
          auto J_acc = J.access();

          /*
              x - component
          */
          J_acc(ix_min, iy_min, cur::jx1)     += jx_0_0;
          J_acc(ix_min, iy_min + 1, cur::jx1) += jx_0_1;
          J_acc(ix_min, iy_min + 2, cur::jx1) += jx_0_2;
          J_acc(ix_min, iy_min + 3, cur::jx1) += jx_0_3;

          J_acc(ix_min + 1, iy_min, cur::jx1)     += jx_1_0;
          J_acc(ix_min + 1, iy_min + 1, cur::jx1) += jx_1_1;
          J_acc(ix_min + 1, iy_min + 2, cur::jx1) += jx_1_2;
          J_acc(ix_min + 1, iy_min + 3, cur::jx1) += jx_1_3;

          J_acc(ix_min + 2, iy_min, cur::jx1)     += jx_2_0;
          J_acc(ix_min + 2, iy_min + 1, cur::jx1) += jx_2_1;
          J_acc(ix_min + 2, iy_min + 2, cur::jx1) += jx_2_2;
          J_acc(ix_min + 2, iy_min + 3, cur::jx1) += jx_2_3;

          if (update_x3) {
            J_acc(ix_min + 3, iy_min, cur::jx1)     += jx_3_0;
            J_acc(ix_min + 3, iy_min + 1, cur::jx1) += jx_3_1;
            J_acc(ix_min + 3, iy_min + 2, cur::jx1) += jx_3_2;
            J_acc(ix_min + 3, iy_min + 3, cur::jx1) += jx_3_3;
          }

          if (update_y3) {
            J_acc(ix_min, iy_min + 4, cur::jx1)     += jx_0_4;
            J_acc(ix_min + 1, iy_min + 4, cur::jx1) += jx_1_4;
            J_acc(ix_min + 2, iy_min + 4, cur::jx1) += jx_2_4;
          }

          if (update_x3 && update_y3) {
            J_acc(ix_min + 3, iy_min + 4, cur::jx1) += jx_3_4;
          }

          /*
              y - component
          */
          J_acc(ix_min, iy_min, cur::jx2)     += jy_0_0;
          J_acc(ix_min + 1, iy_min, cur::jx2) += jy_1_0;
          J_acc(ix_min + 2, iy_min, cur::jx2) += jy_2_0;
          J_acc(ix_min + 3, iy_min, cur::jx2) += jy_3_0;

          J_acc(ix_min, iy_min + 1, cur::jx2)     += jy_0_1;
          J_acc(ix_min + 1, iy_min + 1, cur::jx2) += jy_1_1;
          J_acc(ix_min + 2, iy_min + 1, cur::jx2) += jy_2_1;
          J_acc(ix_min + 3, iy_min + 1, cur::jx2) += jy_3_1;

          J_acc(ix_min, iy_min + 2, cur::jx2)     += jy_0_2;
          J_acc(ix_min + 1, iy_min + 2, cur::jx2) += jy_1_2;
          J_acc(ix_min + 2, iy_min + 2, cur::jx2) += jy_2_2;
          J_acc(ix_min + 3, iy_min + 2, cur::jx2) += jy_3_2;

          if (update_x3) {
            J_acc(ix_min + 4, iy_min, cur::jx2)     += jy_4_0;
            J_acc(ix_min + 4, iy_min + 1, cur::jx2) += jy_4_1;
            J_acc(ix_min + 4, iy_min + 2, cur::jx2) += jy_4_2;
          }

          if (update_y3) {
            J_acc(ix_min, iy_min + 3, cur::jx2)     += jy_0_3;
            J_acc(ix_min + 1, iy_min + 3, cur::jx2) += jy_1_3;
            J_acc(ix_min + 2, iy_min + 3, cur::jx2) += jy_2_3;
            J_acc(ix_min + 3, iy_min + 3, cur::jx2) += jy_3_3;
          }

          if (update_x3 && update_y3) {
            J_acc(ix_min + 4, iy_min + 3, cur::jx2) += jy_4_3;
          }
          /*
              z - component, simulated direction
          */
          J_acc(ix_min, iy_min, cur::jx3)     += QVz * Wz_0_0;
          J_acc(ix_min, iy_min + 1, cur::jx3) += QVz * Wz_0_1;
          J_acc(ix_min, iy_min + 2, cur::jx3) += QVz * Wz_0_2;
          J_acc(ix_min, iy_min + 3, cur::jx3) += QVz * Wz_0_3;

          J_acc(ix_min + 1, iy_min, cur::jx3)     += QVz * Wz_1_0;
          J_acc(ix_min + 1, iy_min + 1, cur::jx3) += QVz * Wz_1_1;
          J_acc(ix_min + 1, iy_min + 2, cur::jx3) += QVz * Wz_1_2;
          J_acc(ix_min + 1, iy_min + 3, cur::jx3) += QVz * Wz_1_3;

          J_acc(ix_min + 2, iy_min, cur::jx3)     += QVz * Wz_2_0;
          J_acc(ix_min + 2, iy_min + 1, cur::jx3) += QVz * Wz_2_1;
          J_acc(ix_min + 2, iy_min + 2, cur::jx3) += QVz * Wz_2_2;
          J_acc(ix_min + 2, iy_min + 3, cur::jx3) += QVz * Wz_2_3;

          J_acc(ix_min + 3, iy_min, cur::jx3)     += QVz * Wz_3_0;
          J_acc(ix_min + 3, iy_min + 1, cur::jx3) += QVz * Wz_3_1;
          J_acc(ix_min + 3, iy_min + 2, cur::jx3) += QVz * Wz_3_2;
          J_acc(ix_min + 3, iy_min + 3, cur::jx3) += QVz * Wz_3_3;

          if (update_x3) {
            J_acc(ix_min + 4, iy_min, cur::jx3)     += QVz * Wz_4_0;
            J_acc(ix_min + 4, iy_min + 1, cur::jx3) += QVz * Wz_4_1;
            J_acc(ix_min + 4, iy_min + 2, cur::jx3) += QVz * Wz_4_2;
            J_acc(ix_min + 4, iy_min + 3, cur::jx3) += QVz * Wz_4_3;
          }

          if (update_y3) {
            J_acc(ix_min, iy_min + 4, cur::jx3)     += QVz * Wz_0_4;
            J_acc(ix_min + 1, iy_min + 4, cur::jx3) += QVz * Wz_1_4;
            J_acc(ix_min + 2, iy_min + 4, cur::jx3) += QVz * Wz_2_4;
            J_acc(ix_min + 3, iy_min + 4, cur::jx3) += QVz * Wz_3_4;
          }
          if (update_x3 && update_y3) {
            J_acc(ix_min + 4, iy_min + 4, cur::jx3) += QVz * Wz_4_4;
          }

        } // dim -> ToDo: 3D!

      } else if constexpr ((O > 3u) && (O < 5u)) {

        // shape function in dim1 -> always required
        real_t   iS_x1[O + 2], fS_x1[O + 2];
        // indices of the shape function
        ncells_t i1_min;

        // call shape function
        prtl_shape::for_deposit<O>(i1_prev(p),
                                   static_cast<real_t>(dx1_prev(p)),
                                   i1(p),
                                   static_cast<real_t>(dx1(p)),
                                   i1_min,
                                   iS_x1,
                                   fS_x1);

        if constexpr (D == Dim::_1D) {
          // ToDo
        } else if constexpr (D == Dim::_2D) {

          // shape function in dim1 -> always required
          real_t   iS_x2[O + 2], fS_x2[O + 2];
          // indices of the shape function
          ncells_t i2_min;

          // call shape function
          prtl_shape::for_deposit<O>(i2_prev(p),
                                            static_cast<real_t>(dx2_prev(p)),
                                            i2(p),
                                            static_cast<real_t>(dx2(p)),
                                            i2_min,
                                            iS_x2,
                                            fS_x2);

          // define weight tensors
          real_t Wx[O + 2][O + 2];
          real_t Wy[O + 2][O + 2];
          real_t Wz[O + 2][O + 2];

// Calculate weight function
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              // Esirkepov 2001, Eq. 38
              Wx[i][j] = (fS_x1[i] - iS_x1[i]) *
                         (iS_x2[j] + HALF * (fS_x2[j] - iS_x2[j]));

              Wy[i][j] = (fS_x2[j] - iS_x2[j]) *
                         (iS_x2[j] + HALF * (fS_x1[i] - iS_x1[i]));

              Wz[i][j] = iS_x1[i] * iS_x2[j] +
                         HALF * (fS_x1[i] - fS_x1[i]) * iS_x2[j] +
                         HALF * iS_x1[i] * (fS_x2[j] - iS_x2[j]) +
                         THIRD * (fS_x1[i] - iS_x1[i]) * (fS_x2[j] - iS_x2[j]);
            }
          }

          // contribution within the shape function stencil
          real_t jx[O + 2][O + 2], jy[O + 2][O + 2], jz[O + 2][O + 2];

          // prefactors for j update
          const real_t Qdx1dt = -coeff * inv_dt;
          const real_t Qdx2dt = -coeff * inv_dt;
          const real_t QVx3   = coeff * vp[2];

          // Calculate current contribution

          // jx
#pragma unroll
          for (int j = 0; j < O + 2; ++j) {
            jx[0][j] = Wx[0][j];
          }

#pragma unroll
          for (int i = 1; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              jx[i][j] = jx[i - 1][j] + Wx[i][j];
            }
          }

          // jy
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
            jy[i][0] = Wy[i][0];
          }

#pragma unroll
          for (int j = 1; j < O + 2; ++j) {
#pragma unroll
            for (int i = 0; i < O + 2; ++i) {
              jy[i][j] = jy[i][j - 1] + Wy[i][j];
            }
          }

          // jz
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              jz[i][j] = Wz[i][j];
            }
          }

          // account for ghost cells
          i1_min += N_GHOSTS;
          i2_min += N_GHOSTS;

          /*
              Current update
          */
          auto J_acc = J.access();

#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              J_acc(i1_min + i, i2_min + j, cur::jx1) += Qdx1dt * jx[i][j];
              J_acc(i1_min + i, i2_min + j, cur::jx2) += Qdx2dt * jy[i][j];
              J_acc(i1_min + i, i2_min + j, cur::jx3) += QVx3 * jz[i][j];
            }
          }

        } else if constexpr (D == Dim::_3D) {
          // shape function in dim2
          real_t   iS_x2[O + 2], fS_x2[O + 2];
          // indices of the shape function
          ncells_t i2_min;
          // call shape function
          prtl_shape::for_deposit<O>(i2_prev(p),
                                     static_cast<real_t>(dx2_prev(p)),
                                     i2(p),
                                     static_cast<real_t>(dx2(p)),
                                     i2_min,
                                     iS_x2,
                                     fS_x2);

          // shape function in dim3
          real_t   iS_x3[O + 2], fS_x3[O + 2];
          // indices of the shape function
          ncells_t i3_min;

          // call shape function
          prtl_shape::for_deposit<O>(i3_prev(p),
                                     static_cast<real_t>(dx3_prev(p)),
                                     i3(p),
                                     static_cast<real_t>(dx3(p)),
                                     i3_min,
                                     iS_x3,
                                     fS_x3);

          // define weight tensors
          real_t Wx[O + 1][O + 1][O + 1];
          real_t Wy[O + 1][O + 1][O + 1];
          real_t Wz[O + 1][O + 1][O + 1];

// Calculate weight function
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
#pragma unroll
              for (int k = 0; k < O + 2; ++k) {
                // Esirkepov 2001, Eq. 31
                Wx[i][j][k] = THIRD * (fS_x1[i] - iS_x1[i]) *
                              ((iS_x2[j] * iS_x3[k] + fS_x2[j] * fS_x3[k]) +
                               HALF * (iS_x3[k] * fS_x2[j] + iS_x2[j] * fS_x3[k]));

                Wy[i][j][k] = THIRD * (fS_x2[j] - iS_x2[j]) *
                              (iS_x1[i] * iS_x3[k] + fS_x1[i] * fS_x3[k] +
                               HALF * (iS_x3[k] * fS_x1[i] + iS_x1[i] * fS_x3[k]));

                Wz[i][j][k] = THIRD * (fS_x3[k] - iS_x3[k]) *
                              (iS_x1[i] * iS_x2[j] + fS_x1[i] * fS_x2[j] +
                               HALF * (iS_x1[i] * fS_x2[j] + iS_x2[j] * fS_x1[i]));
              }
            }
          }

          // contribution within the shape function stencil
          real_t jx[O + 2][O + 2][O + 2], jy[O + 2][O + 2][O + 2],
            jz[O + 2][O + 2][O + 2];

          // prefactors to j update
          const real_t Qdxdt = coeff * inv_dt;
          const real_t Qdydt = coeff * inv_dt;
          const real_t Qdzdt = coeff * inv_dt;

          // Calculate current contribution

          // jx
#pragma unroll
          for (int j = 0; j < O + 2; ++j) {
#pragma unroll
            for (int k = 0; k < O + 2; ++k) {
              jx[0][j][k] = -Qdxdt * Wx[0][j][k];
            }
          }

#pragma unroll
          for (int i = 1; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
#pragma unroll
              for (int k = 0; j < O + 2; ++k) {
                jx[i][j][k] = jx[i - 1][j][k] - Qdxdt * Wx[i][j][k];
              }
            }
          }

          // jy
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int k = 0; k < O + 2; ++k) {
              jy[i][0][k] = -Qdydt * Wy[i][0][k];
            }
          }

#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 1; j < O + 2; ++j) {
#pragma unroll
              for (int k = 0; k < O + 2; ++k) {
                jy[i][j][k] = jy[i][j - 1][k] - Qdydt * Wy[i][j][k];
              }
            }
          }

          // jz
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              jy[i][j][0] = -Qdydt * Wy[i][j][0];
            }
          }

#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
#pragma unroll
              for (int k = 1; k < O + 2; ++k) {
                jz[i][j][k] = jz[i][j][k - 1] - Qdzdt * Wz[i][j][k];
              }
            }
          }

          /*
            Current update
          */
          auto J_acc = J.access();

#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
#pragma unroll
              for (int k = 1; k < O + 2; ++k) {
                J_acc(i1_min + i, i2_min + j, i3_min, cur::jx1) += jx[i][j][k];
                J_acc(i1_min + i, i2_min + j, i3_min, cur::jx2) += jy[i][j][k];
                J_acc(i1_min + i, i2_min + j, i3_min, cur::jx3) += jz[i][j][k];
              }
            }
          }
        }

      } else { // order
        raise::KernelError(HERE, "Unsupported interpolation order");
      }
    }
  };
} // namespace kernel

#undef i_di_to_Xi

#endif // KERNELS_CURRENTS_DEPOSIT_HPP
