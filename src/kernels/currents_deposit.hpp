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
#include "utils/numeric.h"

#include <Kokkos_Core.hpp>

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))

namespace kernel {
  using namespace ntt;

  /**
   * @brief Algorithm for the current deposition
   */
  template <SimEngine::type S, class M>
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
                           real_t                         dt)
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
      , inv_dt { ONE / dt } {}

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

      const auto dxp_r_1 { static_cast<prtldx_t>(i1(p) == i1_prev(p)) *
                           (dx1(p) + dx1_prev(p)) * static_cast<prtldx_t>(INV_2) };

      const real_t Wx1_1 { INV_2 * (dxp_r_1 + dx1_prev(p) +
                                    static_cast<real_t>(i1(p) > i1_prev(p))) };
      const real_t Wx1_2 { INV_2 * (dx1(p) + dxp_r_1 +
                                    static_cast<real_t>(
                                      static_cast<int>(i1(p) > i1_prev(p)) +
                                      i1_prev(p) - i1(p))) };
      const real_t Fx1_1 { (static_cast<real_t>(i1(p) > i1_prev(p)) + dxp_r_1 -
                            dx1_prev(p)) *
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

          J_acc(i1(p) + N_GHOSTS, i2(p) + N_GHOSTS, cur::jx3) += Fx3_2 *
                                                                 (ONE - Wx1_2) *
                                                                 (ONE - Wx2_2);
          J_acc(i1(p) + N_GHOSTS + 1,
                i2(p) + N_GHOSTS,
                cur::jx3) += Fx3_2 * Wx1_2 * (ONE - Wx2_2);
          J_acc(i1(p) + N_GHOSTS,
                i2(p) + N_GHOSTS + 1,
                cur::jx3) += Fx3_2 * (ONE - Wx1_2) * Wx2_2;
          J_acc(i1(p) + N_GHOSTS + 1, i2(p) + N_GHOSTS + 1, cur::jx3) += Fx3_2 *
                                                                         Wx1_2 *
                                                                         Wx2_2;
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
    }
  };

} // namespace kernel

#undef i_di_to_Xi

#endif // KERNELS_CURRENTS_DEPOSIT_HPP
