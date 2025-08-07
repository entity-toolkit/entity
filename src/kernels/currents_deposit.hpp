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
          coord_t<Dim::_2D> xp_ { ZERO };
          xp_[0] = xp[0];
          real_t       theta_Cd { xp[1] };
          const real_t theta_Ph { metric.template convert<2, Crd::Cd, Crd::Ph>(
            theta_Cd) };
          const real_t small_angle { static_cast<real_t>(constant::SMALL_ANGLE_GR) };
          const auto large_angle { static_cast<real_t>(constant::PI) - small_angle };
          if (theta_Ph < small_angle) {
            theta_Cd = metric.template convert<2, Crd::Ph, Crd::Cd>(small_angle);
          } else if (theta_Ph >= large_angle) {
            theta_Cd = metric.template convert<2, Crd::Ph, Crd::Cd>(large_angle);
          }
          xp_[1] = theta_Cd;
          metric.template transform<Idx::D, Idx::U>(xp_,
                                                    { ux1(p), ux2(p), ux3(p) },
                                                    vp);
          inv_energy = metric.alpha(xp_) /
                       math::sqrt(ONE + ux1(p) * vp[0] + ux2(p) * vp[1] +
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
      if constexpr (O == 0u) {
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
                  cur::jx3) += Fx3_1 * Wx1_1 * (ONE - Wx2_1);
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
      } else if constexpr ((O >= 1u) and (O <= 5u)) {

        // shape function in dim1 -> always required
        real_t iS_x1[O + 2], fS_x1[O + 2];
        // indices of the shape function
        int    i1_min, i1_max;

        // call shape function
        prtl_shape::for_deposit<O>(i1_prev(p),
                                   static_cast<real_t>(dx1_prev(p)),
                                   i1(p),
                                   static_cast<real_t>(dx1(p)),
                                   i1_min,
                                   i1_max,
                                   iS_x1,
                                   fS_x1);

        if constexpr (D == Dim::_1D) {
          // ToDo
          raise::KernelNotImplementedError(HERE);
        } else if constexpr (D == Dim::_2D) {

          // shape function in dim1 -> always required
          real_t iS_x2[O + 2], fS_x2[O + 2];
          // indices of the shape function
          int    i2_min, i2_max;

          // call shape function
          prtl_shape::for_deposit<O>(i2_prev(p),
                                     static_cast<real_t>(dx2_prev(p)),
                                     i2(p),
                                     static_cast<real_t>(dx2(p)),
                                     i2_min,
                                     i2_max,
                                     iS_x2,
                                     fS_x2);

          // define weight tensors
          real_t Wx1[O + 2][O + 2];
          real_t Wx2[O + 2][O + 2];
          real_t Wx3[O + 2][O + 2];

// Calculate weight function
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              // Esirkepov 2001, Eq. 38
              Wx1[i][j] = (fS_x1[i] - iS_x1[i]) *
                          (iS_x2[j] + HALF * (fS_x2[j] - iS_x2[j]));

              Wx2[i][j] = (fS_x2[j] - iS_x2[j]) *
                          (iS_x2[j] + HALF * (fS_x1[i] - iS_x1[i]));

              Wx3[i][j] = iS_x1[i] * iS_x2[j] +
                          HALF * (fS_x1[i] - fS_x1[i]) * iS_x2[j] +
                          HALF * iS_x1[i] * (fS_x2[j] - iS_x2[j]) +
                          THIRD * (fS_x1[i] - iS_x1[i]) * (fS_x2[j] - iS_x2[j]);

              // Wx1[i][j] = HALF * (fS_x1[i] - iS_x1[i]) * (fS_x2[j] + iS_x2[j]);

              // Wx2[i][j] = HALF * (fS_x1[i] + iS_x1[i]) * (fS_x2[j] - iS_x2[j]);

              // Wx3[i][j] = THIRD * (fS_x2[j] * (HALF * iS_x1[i] + fS_x2[j]) +
              //                     iS_x2[j] * (HALF * fS_x2[j] + iS_x2[i]));
            }
          }

          // contribution within the shape function stencil
          real_t jx1[O + 2][O + 2], jx2[O + 2][O + 2], jx3[O + 2][O + 2];

          // prefactors for j update
          const real_t Qdx1dt = -coeff * inv_dt;
          const real_t Qdx2dt = -coeff * inv_dt;
          const real_t QVx3   = coeff * vp[2];

          // Calculate current contribution

          // jx1
#pragma unroll
          for (int j = 0; j < O + 2; ++j) {
            jx1[0][j] = Wx1[0][j];
          }

#pragma unroll
          for (int i = 1; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              jx1[i][j] = jx1[i - 1][j] + Wx1[i][j];
            }
          }

          // jx2
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
            jx2[i][0] = Wx2[i][0];
          }

#pragma unroll
          for (int j = 1; j < O + 2; ++j) {
#pragma unroll
            for (int i = 0; i < O + 2; ++i) {
              jx2[i][j] = jx2[i][j - 1] + Wx2[i][j];
            }
          }

          // jx3
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              jx3[i][j] = Wx3[i][j];
            }
          }

          // account for ghost cells
          i1_min += N_GHOSTS;
          i2_min += N_GHOSTS;
          i1_max += N_GHOSTS;
          i2_max += N_GHOSTS;

          // get number of update indices for asymmetric movement
          const int di_x1 = i1_max - i1_min;
          const int di_x2 = i2_max - i2_min;

          /*
              Current update
          */
          auto J_acc = J.access();

          for (int i = 0; i <= di_x1; ++i) {
            for (int j = 0; j <= di_x2; ++j) {
              J_acc(i1_min + i, i2_min + j, cur::jx1) += Qdx1dt * jx1[i][j];
            }
          }

          for (int i = 0; i <= di_x1; ++i) {
            for (int j = 0; j <= di_x2; ++j) {
              J_acc(i1_min + i, i2_min + j, cur::jx2) += Qdx2dt * jx2[i][j];
            }
          }

          for (int i = 0; i <= di_x1; ++i) {
            for (int j = 0; j <= di_x2; ++j) {
              J_acc(i1_min + i, i2_min + j, cur::jx3) += QVx3 * jx3[i][j];
            }
          }

        } else if constexpr (D == Dim::_3D) {
          // shape function in dim2
          real_t iS_x2[O + 2], fS_x2[O + 2];
          // indices of the shape function
          int    i2_min, i2_max;
          // call shape function
          prtl_shape::for_deposit<O>(i2_prev(p),
                                     static_cast<real_t>(dx2_prev(p)),
                                     i2(p),
                                     static_cast<real_t>(dx2(p)),
                                     i2_min,
                                     i2_max,
                                     iS_x2,
                                     fS_x2);

          // shape function in dim3
          real_t iS_x3[O + 2], fS_x3[O + 2];
          // indices of the shape function
          int    i3_min, i3_max;

          // call shape function
          prtl_shape::for_deposit<O>(i3_prev(p),
                                     static_cast<real_t>(dx3_prev(p)),
                                     i3(p),
                                     static_cast<real_t>(dx3(p)),
                                     i3_min,
                                     i3_max,
                                     iS_x3,
                                     fS_x3);

          // define weight tensors
          real_t Wx1[O + 2][O + 2][O + 2];
          real_t Wx2[O + 2][O + 2][O + 2];
          real_t Wx3[O + 2][O + 2][O + 2];

// Calculate weight function
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
#pragma unroll
              for (int k = 0; k < O + 2; ++k) {
                // Esirkepov 2001, Eq. 31
                Wx1[i][j][k] = THIRD * (fS_x1[i] - iS_x1[i]) *
                               ((iS_x2[j] * iS_x3[k] + fS_x2[j] * fS_x3[k]) +
                                HALF * (iS_x3[k] * fS_x2[j] + iS_x2[j] * fS_x3[k]));

                Wx2[i][j][k] = THIRD * (fS_x2[j] - iS_x2[j]) *
                               (iS_x1[i] * iS_x3[k] + fS_x1[i] * fS_x3[k] +
                                HALF * (iS_x3[k] * fS_x1[i] + iS_x1[i] * fS_x3[k]));

                Wx3[i][j][k] = THIRD * (fS_x3[k] - iS_x3[k]) *
                               (iS_x1[i] * iS_x2[j] + fS_x1[i] * fS_x2[j] +
                                HALF * (iS_x1[i] * fS_x2[j] + iS_x2[j] * fS_x1[i]));
              }
            }
          }

          // contribution within the shape function stencil
          real_t jx1[O + 2][O + 2][O + 2], jx2[O + 2][O + 2][O + 2],
            jx3[O + 2][O + 2][O + 2];

          // prefactors to j update
          const real_t Qdxdt = coeff * inv_dt;
          const real_t Qdydt = coeff * inv_dt;
          const real_t Qdzdt = coeff * inv_dt;

          // Calculate current contribution

          // jx1
#pragma unroll
          for (int j = 0; j < O + 2; ++j) {
#pragma unroll
            for (int k = 0; k < O + 2; ++k) {
              jx1[0][j][k] = -Qdxdt * Wx1[0][j][k];
            }
          }

#pragma unroll
          for (int i = 1; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
#pragma unroll
              for (int k = 0; j < O + 2; ++k) {
                jx1[i][j][k] = jx1[i - 1][j][k] - Qdxdt * Wx1[i][j][k];
              }
            }
          }

          // jx2
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int k = 0; k < O + 2; ++k) {
              jx2[i][0][k] = -Qdydt * Wx2[i][0][k];
            }
          }

#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 1; j < O + 2; ++j) {
#pragma unroll
              for (int k = 0; k < O + 2; ++k) {
                jx2[i][j][k] = jx2[i][j - 1][k] - Qdydt * Wx2[i][j][k];
              }
            }
          }

          // jx3
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              jx2[i][j][0] = -Qdydt * Wx2[i][j][0];
            }
          }

#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
#pragma unroll
              for (int k = 1; k < O + 2; ++k) {
                jx3[i][j][k] = jx3[i][j][k - 1] - Qdzdt * Wx3[i][j][k];
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
                J_acc(i1_min + i, i2_min + j, i3_min, cur::jx1) += jx1[i][j][k];
                J_acc(i1_min + i, i2_min + j, i3_min, cur::jx2) += jx2[i][j][k];
                J_acc(i1_min + i, i2_min + j, i3_min, cur::jx3) += jx3[i][j][k];
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
//
//   } else if constexpr (O == 2u) {
//     /*
//      * Higher order charge conserving current deposition based on
//      * Esirkepov (2001) https://ui.adsabs.harvard.edu/abs/2001CoPhC.135..144E/abstract
//      **/

//     // iS -> shape function for init position
//     // fS -> shape function for final position

//     // shape function at integer points (one coeff is always ZERO)
//     int    i1_min;
//     real_t iS_x1_0, iS_x1_1, iS_x1_2, iS_x1_3;
//     real_t fS_x1_0, fS_x1_1, fS_x1_2, fS_x1_3;

//     // clang-format off
//     prtl_shape::for_deposit_2nd(i1_prev(p), static_cast<real_t>(dx1_prev(p)),
//                                 i1(p), static_cast<real_t>(dx1(p)),
//                                 i1_min,
//                                 iS_x1_0, iS_x1_1, iS_x1_2, iS_x1_3,
//                                 fS_x1_0, fS_x1_1, fS_x1_2, fS_x1_3);
//     // clang-format on

//     if constexpr (D == Dim::_1D) {
//       raise::KernelNotImplementedError(HERE);
//     } else if constexpr (D == Dim::_2D) {

//       // shape function at integer points (one coeff is always ZERO)
//       int    i2_min;
//       real_t iS_x2_0, iS_x2_1, iS_x2_2, iS_x2_3;
//       real_t fS_x2_0, fS_x2_1, fS_x2_2, fS_x2_3;

//       // clang-format off
//       prtl_shape::for_deposit_2nd(i2_prev(p), static_cast<real_t>(dx2_prev(p)),
//                                   i2(p), static_cast<real_t>(dx2(p)),
//                                   i2_min,
//                                   iS_x2_0, iS_x2_1, iS_x2_2, iS_x2_3,
//                                   fS_x2_0, fS_x2_1, fS_x2_2, fS_x2_3);
//       // clang-format on
//       // x1-components
//       const auto Wx1_00 = HALF * (fS_x1_0 - iS_x1_0) * (fS_x2_0 + iS_x2_0);
//       const auto Wx1_01 = HALF * (fS_x1_0 - iS_x1_0) * (fS_x2_1 + iS_x2_1);
//       const auto Wx1_02 = HALF * (fS_x1_0 - iS_x1_0) * (fS_x2_2 + iS_x2_2);
//       const auto Wx1_03 = HALF * (fS_x1_0 - iS_x1_0) * (fS_x2_3 + iS_x2_3);

//       const auto Wx1_10 = HALF * (fS_x1_1 - iS_x1_1) * (fS_x2_0 + iS_x2_0);
//       const auto Wx1_11 = HALF * (fS_x1_1 - iS_x1_1) * (fS_x2_1 + iS_x2_1);
//       const auto Wx1_12 = HALF * (fS_x1_1 - iS_x1_1) * (fS_x2_2 + iS_x2_2);
//       const auto Wx1_13 = HALF * (fS_x1_1 - iS_x1_1) * (fS_x2_3 + iS_x2_3);

//       const auto Wx1_20 = HALF * (fS_x1_2 - iS_x1_2) * (fS_x2_0 + iS_x2_0);
//       const auto Wx1_21 = HALF * (fS_x1_2 - iS_x1_2) * (fS_x2_1 + iS_x2_1);
//       const auto Wx1_22 = HALF * (fS_x1_2 - iS_x1_2) * (fS_x2_2 + iS_x2_2);
//       const auto Wx1_23 = HALF * (fS_x1_2 - iS_x1_2) * (fS_x2_3 + iS_x2_3);

//       const auto Wx1_30 = HALF * (fS_x1_3 - iS_x1_3) * (fS_x2_0 + iS_x2_0);
//       const auto Wx1_31 = HALF * (fS_x1_3 - iS_x1_3) * (fS_x2_1 + iS_x2_1);
//       const auto Wx1_32 = HALF * (fS_x1_3 - iS_x1_3) * (fS_x2_2 + iS_x2_2);
//       const auto Wx1_33 = HALF * (fS_x1_3 - iS_x1_3) * (fS_x2_3 + iS_x2_3);

//       // x2-components
//       const auto Wx2_00 = HALF * (fS_x1_0 + iS_x1_0) * (fS_x2_0 - iS_x2_0);
//       const auto Wx2_01 = HALF * (fS_x1_0 + iS_x1_0) * (fS_x2_1 - iS_x2_1);
//       const auto Wx2_02 = HALF * (fS_x1_0 + iS_x1_0) * (fS_x2_2 - iS_x2_2);
//       const auto Wx2_03 = HALF * (fS_x1_0 + iS_x1_0) * (fS_x2_3 - iS_x2_3);

//       const auto Wx2_10 = HALF * (fS_x1_1 + iS_x1_1) * (fS_x2_0 - iS_x2_0);
//       const auto Wx2_11 = HALF * (fS_x1_1 + iS_x1_1) * (fS_x2_1 - iS_x2_1);
//       const auto Wx2_12 = HALF * (fS_x1_1 + iS_x1_1) * (fS_x2_2 - iS_x2_2);
//       const auto Wx2_13 = HALF * (fS_x1_1 + iS_x1_1) * (fS_x2_3 - iS_x2_3);

//       const auto Wx2_20 = HALF * (fS_x1_2 + iS_x1_2) * (fS_x2_0 - iS_x2_0);
//       const auto Wx2_21 = HALF * (fS_x1_2 + iS_x1_2) * (fS_x2_1 - iS_x2_1);
//       const auto Wx2_22 = HALF * (fS_x1_2 + iS_x1_2) * (fS_x2_2 - iS_x2_2);
//       const auto Wx2_23 = HALF * (fS_x1_2 + iS_x1_2) * (fS_x2_3 - iS_x2_3);

//       const auto Wx2_30 = HALF * (fS_x1_3 + iS_x1_3) * (fS_x2_0 - iS_x2_0);
//       const auto Wx2_31 = HALF * (fS_x1_3 + iS_x1_3) * (fS_x2_1 - iS_x2_1);
//       const auto Wx2_32 = HALF * (fS_x1_3 + iS_x1_3) * (fS_x2_2 - iS_x2_2);
//       const auto Wx2_33 = HALF * (fS_x1_3 + iS_x1_3) * (fS_x2_3 - iS_x2_3);

//       // x3-components
//       const auto Wx3_00 = THIRD * (fS_x2_0 * (HALF * iS_x1_0 + fS_x1_0) +
//                                    iS_x2_0 * (HALF * fS_x1_0 + iS_x1_0));
//       const auto Wx3_01 = THIRD * (fS_x2_1 * (HALF * iS_x1_0 + fS_x1_0) +
//                                    iS_x2_1 * (HALF * fS_x1_0 + iS_x1_0));
//       const auto Wx3_02 = THIRD * (fS_x2_2 * (HALF * iS_x1_0 + fS_x1_0) +
//                                    iS_x2_2 * (HALF * fS_x1_0 + iS_x1_0));
//       const auto Wx3_03 = THIRD * (fS_x2_3 * (HALF * iS_x1_0 + fS_x1_0) +
//                                    iS_x2_3 * (HALF * fS_x1_0 + iS_x1_0));

//       const auto Wx3_10 = THIRD * (fS_x2_0 * (HALF * iS_x1_1 + fS_x1_1) +
//                                    iS_x2_0 * (HALF * fS_x1_1 + iS_x1_1));
//       const auto Wx3_11 = THIRD * (fS_x2_1 * (HALF * iS_x1_1 + fS_x1_1) +
//                                    iS_x2_1 * (HALF * fS_x1_1 + iS_x1_1));
//       const auto Wx3_12 = THIRD * (fS_x2_2 * (HALF * iS_x1_1 + fS_x1_1) +
//                                    iS_x2_2 * (HALF * fS_x1_1 + iS_x1_1));
//       const auto Wx3_13 = THIRD * (fS_x2_3 * (HALF * iS_x1_1 + fS_x1_1) +
//                                    iS_x2_3 * (HALF * fS_x1_1 + iS_x1_1));

//       const auto Wx3_20 = THIRD * (fS_x2_0 * (HALF * iS_x1_2 + fS_x1_2) +
//                                    iS_x2_0 * (HALF * fS_x1_2 + iS_x1_2));
//       const auto Wx3_21 = THIRD * (fS_x2_1 * (HALF * iS_x1_2 + fS_x1_2) +
//                                    iS_x2_1 * (HALF * fS_x1_2 + iS_x1_2));
//       const auto Wx3_22 = THIRD * (fS_x2_2 * (HALF * iS_x1_2 + fS_x1_2) +
//                                    iS_x2_2 * (HALF * fS_x1_2 + iS_x1_2));
//       const auto Wx3_23 = THIRD * (fS_x2_3 * (HALF * iS_x1_2 + fS_x1_2) +
//                                    iS_x2_3 * (HALF * fS_x1_2 + iS_x1_2));

//       const auto Wx3_30 = THIRD * (fS_x2_0 * (HALF * iS_x1_3 + fS_x1_3) +
//                                    iS_x2_0 * (HALF * fS_x1_3 + iS_x1_3));
//       const auto Wx3_31 = THIRD * (fS_x2_1 * (HALF * iS_x1_3 + fS_x1_3) +
//                                    iS_x2_1 * (HALF * fS_x1_3 + iS_x1_3));
//       const auto Wx3_32 = THIRD * (fS_x2_2 * (HALF * iS_x1_3 + fS_x1_3) +
//                                    iS_x2_2 * (HALF * fS_x1_3 + iS_x1_3));
//       const auto Wx3_33 = THIRD * (fS_x2_3 * (HALF * iS_x1_3 + fS_x1_3) +
//                                    iS_x2_3 * (HALF * fS_x1_3 + iS_x1_3));

//       // x1-component
//       const auto jx1_00 = Wx1_00;
//       const auto jx1_10 = jx1_00 + Wx1_10;
//       const auto jx1_20 = jx1_10 + Wx1_20;
//       const auto jx1_30 = jx1_20 + Wx1_30;

//       const auto jx1_01 = Wx1_01;
//       const auto jx1_11 = jx1_01 + Wx1_11;
//       const auto jx1_21 = jx1_11 + Wx1_21;
//       const auto jx1_31 = jx1_21 + Wx1_31;

//       const auto jx1_02 = Wx1_02;
//       const auto jx1_12 = jx1_02 + Wx1_12;
//       const auto jx1_22 = jx1_12 + Wx1_22;
//       const auto jx1_32 = jx1_22 + Wx1_32;

//       const auto jx1_03 = Wx1_03;
//       const auto jx1_13 = jx1_03 + Wx1_13;
//       const auto jx1_23 = jx1_13 + Wx1_23;
//       const auto jx1_33 = jx1_23 + Wx1_33;

//       // y-component
//       const auto jx2_00 = Wx2_00;
//       const auto jx2_01 = jx2_00 + Wx2_01;
//       const auto jx2_02 = jx2_01 + Wx2_02;
//       const auto jx2_03 = jx2_02 + Wx2_03;

//       const auto jx2_10 = Wx2_10;
//       const auto jx2_11 = jx2_10 + Wx2_11;
//       const auto jx2_12 = jx2_11 + Wx2_12;
//       const auto jx2_13 = jx2_12 + Wx2_13;

//       const auto jx2_20 = Wx2_20;
//       const auto jx2_21 = jx2_20 + Wx2_21;
//       const auto jx2_22 = jx2_21 + Wx2_22;
//       const auto jx2_23 = jx2_22 + Wx2_23;

//       const auto jx2_30 = Wx2_30;
//       const auto jx2_31 = jx2_30 + Wx2_31;
//       const auto jx2_32 = jx2_31 + Wx2_32;
//       const auto jx2_33 = jx2_32 + Wx2_33;

//       i1_min  += N_GHOSTS;
//       i2_min  += N_GHOSTS;

//       // @TODO: not sure about the signs here
//       const real_t Qdx1dt = -coeff * inv_dt;
//       const real_t Qdx2dt = -coeff * inv_dt;
//       const real_t QVx3   = coeff * vp[2];

//       auto J_acc = J.access();

//       // x1-currents
//       J_acc(i1_min + 0, i2_min + 0, cur::jx1) += Qdx1dt * jx1_00;
//       J_acc(i1_min + 0, i2_min + 1, cur::jx1) += Qdx1dt * jx1_01;
//       J_acc(i1_min + 0, i2_min + 2, cur::jx1) += Qdx1dt * jx1_02;
//       J_acc(i1_min + 0, i2_min + 3, cur::jx1) += Qdx1dt * jx1_03;

//       J_acc(i1_min + 1, i2_min + 0, cur::jx1) += Qdx1dt * jx1_10;
//       J_acc(i1_min + 1, i2_min + 1, cur::jx1) += Qdx1dt * jx1_11;
//       J_acc(i1_min + 1, i2_min + 2, cur::jx1) += Qdx1dt * jx1_12;
//       J_acc(i1_min + 1, i2_min + 3, cur::jx1) += Qdx1dt * jx1_13;

//       J_acc(i1_min + 2, i2_min + 0, cur::jx1) += Qdx1dt * jx1_20;
//       J_acc(i1_min + 2, i2_min + 1, cur::jx1) += Qdx1dt * jx1_21;
//       J_acc(i1_min + 2, i2_min + 2, cur::jx1) += Qdx1dt * jx1_22;
//       J_acc(i1_min + 2, i2_min + 3, cur::jx1) += Qdx1dt * jx1_23;

//       J_acc(i1_min + 3, i2_min + 0, cur::jx1) += Qdx1dt * jx1_30;
//       J_acc(i1_min + 3, i2_min + 1, cur::jx1) += Qdx1dt * jx1_31;
//       J_acc(i1_min + 3, i2_min + 2, cur::jx1) += Qdx1dt * jx1_32;
//       J_acc(i1_min + 3, i2_min + 3, cur::jx1) += Qdx1dt * jx1_33;

//       // x2-currents
//       J_acc(i1_min + 0, i2_min + 0, cur::jx2) += Qdx2dt * jx2_00;
//       J_acc(i1_min + 0, i2_min + 1, cur::jx2) += Qdx2dt * jx2_01;
//       J_acc(i1_min + 0, i2_min + 2, cur::jx2) += Qdx2dt * jx2_02;
//       J_acc(i1_min + 0, i2_min + 3, cur::jx2) += Qdx2dt * jx2_03;

//       J_acc(i1_min + 1, i2_min + 0, cur::jx2) += Qdx2dt * jx2_10;
//       J_acc(i1_min + 1, i2_min + 1, cur::jx2) += Qdx2dt * jx2_11;
//       J_acc(i1_min + 1, i2_min + 2, cur::jx2) += Qdx2dt * jx2_12;
//       J_acc(i1_min + 1, i2_min + 3, cur::jx2) += Qdx2dt * jx2_13;

//       J_acc(i1_min + 2, i2_min + 0, cur::jx2) += Qdx2dt * jx2_20;
//       J_acc(i1_min + 2, i2_min + 1, cur::jx2) += Qdx2dt * jx2_21;
//       J_acc(i1_min + 2, i2_min + 2, cur::jx2) += Qdx2dt * jx2_22;
//       J_acc(i1_min + 2, i2_min + 3, cur::jx2) += Qdx2dt * jx2_23;

//       J_acc(i1_min + 3, i2_min + 0, cur::jx2) += Qdx2dt * jx2_30;
//       J_acc(i1_min + 3, i2_min + 1, cur::jx2) += Qdx2dt * jx2_31;
//       J_acc(i1_min + 3, i2_min + 2, cur::jx2) += Qdx2dt * jx2_32;
//       J_acc(i1_min + 3, i2_min + 3, cur::jx2) += Qdx2dt * jx2_33;

//       // x3-currents
//       J_acc(i1_min + 0, i2_min + 0, cur::jx3) += QVx3 * Wx3_00;
//       J_acc(i1_min + 0, i2_min + 1, cur::jx3) += QVx3 * Wx3_01;
//       J_acc(i1_min + 0, i2_min + 2, cur::jx3) += QVx3 * Wx3_02;
//       J_acc(i1_min + 0, i2_min + 3, cur::jx3) += QVx3 * Wx3_03;

//       J_acc(i1_min + 1, i2_min + 0, cur::jx3) += QVx3 * Wx3_10;
//       J_acc(i1_min + 1, i2_min + 1, cur::jx3) += QVx3 * Wx3_11;
//       J_acc(i1_min + 1, i2_min + 2, cur::jx3) += QVx3 * Wx3_12;
//       J_acc(i1_min + 1, i2_min + 3, cur::jx3) += QVx3 * Wx3_13;

//       J_acc(i1_min + 2, i2_min + 0, cur::jx3) += QVx3 * Wx3_20;
//       J_acc(i1_min + 2, i2_min + 1, cur::jx3) += QVx3 * Wx3_21;
//       J_acc(i1_min + 2, i2_min + 2, cur::jx3) += QVx3 * Wx3_22;
//       J_acc(i1_min + 2, i2_min + 3, cur::jx3) += QVx3 * Wx3_23;

//       J_acc(i1_min + 3, i2_min + 0, cur::jx3) += QVx3 * Wx3_30;
//       J_acc(i1_min + 3, i2_min + 1, cur::jx3) += QVx3 * Wx3_31;
//       J_acc(i1_min + 3, i2_min + 2, cur::jx3) += QVx3 * Wx3_32;
//       J_acc(i1_min + 3, i2_min + 3, cur::jx3) += QVx3 * Wx3_33;

//     } else if constexpr (D == Dim::_3D) {
// raise::KernelNotImplementedError(HERE);
//     } // dimension

#endif // KERNELS_CURRENTS_DEPOSIT_HPP
