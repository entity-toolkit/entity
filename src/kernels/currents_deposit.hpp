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

    Inline void shape_function_2nd(real_t&        S0_0,
                                   real_t&        S0_1,
                                   real_t&        S0_2,
                                   real_t&        S0_3,
                                   real_t&        S1_0,
                                   real_t&        S1_1,
                                   real_t&        S1_2,
                                   real_t&        S1_3,
                                   ncells_t&      i_min,
                                   const index_t& i,
                                   const real_t&  dx,
                                   const index_t& i_prev,
                                   const real_t&  dx_prev) const {
      /*
        Shape function per particle is a 4 element array.
        We need to find which indices are contributing to the shape function
        For this we first compute the indices of the particle position

        Let * be the particle position at the current timestep
        Let x be the particle position at the previous timestep


          (-1)    0      1      2      3
        ___________________________________
        |      |  x*  |  x*  |  x*  |      |   // shift_i = 0
        |______|______|______|______|______|
        |      |  x   |  x*  |  x*  |  *   |   // shift_i = 1
        |______|______|______|______|______|
        |  *   |  x*  |  x*  |  x   |      |   // shift_i = -1
        |______|______|______|______|______|
      */

      // find shift in indices
      const auto dx_less_half = static_cast<int>(dx < static_cast<prtldx_t>(0.5));
      const auto dx_prev_less_half = static_cast<int>(
        dx_prev < static_cast<prtldx_t>(0.5));
      const auto shift_x { (i - i_prev) - (dx_less_half - dx_prev_less_half) };

      const real_t dx_prev_diff = static_cast<real_t>(dx_prev) +
                                  static_cast<real_t>(
                                    dx_prev < static_cast<prtldx_t>(0.5));
      const real_t dx_diff = static_cast<real_t>(dx) +
                             static_cast<real_t>(dx < static_cast<prtldx_t>(0.5));

      // find indices and define shape function
      if (shift_x > 0) {
        /*
            (-1)    0      1      2      3
          ___________________________________
          |      |  x   |  x*  |  x*  |  *   |   // shift_i = 1
          |______|______|______|______|______|
        */
        i_min = i_prev - dx_prev_less_half + N_GHOSTS;

        S0_0 = HALF * SQR(static_cast<real_t>(1.5) - dx_prev_diff);
        S0_1 = static_cast<real_t>(0.75) - SQR(ONE - dx_prev_diff);
        S0_2 = HALF * SQR(HALF - dx_prev_diff);
        S0_3 = ZERO;

        S1_0 = ZERO;
        S1_1 = HALF * SQR(static_cast<real_t>(1.5) - dx_diff);
        S1_2 = static_cast<real_t>(0.75) - SQR(ONE - dx_diff);
        S1_3 = HALF * SQR(HALF - dx_diff);
      } else if (shift_x < 0) {
        /*
            (-1)    0      1      2      3
          ___________________________________
          |  *   |  x*  |  x*  |  x   |      |   // shift_i = -1
          |______|______|______|______|______|
        */
        i_min = i - dx_less_half + N_GHOSTS;

        S0_0 = ZERO;
        S0_1 = HALF * SQR(static_cast<real_t>(1.5) - dx_prev_diff);
        S0_2 = static_cast<real_t>(0.75) - SQR(ONE - dx_prev_diff);
        S0_3 = HALF * SQR(HALF - dx_prev_diff);

        S1_0 = HALF * SQR(static_cast<real_t>(1.5) - dx_diff);
        S1_1 = static_cast<real_t>(0.75) - SQR(ONE - dx_diff);
        S1_2 = HALF * SQR(HALF - dx_diff);
        S1_3 = ZERO;
      } else {
        /*
            (-1)    0      1      2      3
          ___________________________________
          |      |  x*  |  x*  |  x*  |      |   // shift_i = 0
          |______|______|______|______|______|
        */
        i_min = i - dx_less_half + N_GHOSTS;

        S0_0 = HALF * SQR(static_cast<real_t>(1.5) - dx_prev_diff);
        S0_1 = static_cast<real_t>(0.75) - SQR(ONE - dx_prev_diff);
        S0_2 = HALF * SQR(HALF - dx_prev_diff);
        S0_3 = ZERO;

        S1_0 = HALF * SQR(static_cast<real_t>(1.5) - dx_diff);
        S1_1 = static_cast<real_t>(0.75) - SQR(ONE - dx_diff);
        S1_2 = HALF * SQR(HALF - dx_diff);
        S1_3 = ZERO;
      }
    }

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
        real_t   S0x_0, S0x_1, S0x_2, S0x_3;
        // shape function at current timestep
        real_t   S1x_0, S1x_1, S1x_2, S1x_3;
        // indices of the shape function
        ncells_t ix_min;
        // find indices and define shape function
        shape_function_2nd(S0x_0,
                           S0x_1,
                           S0x_2,
                           S0x_3,
                           S1x_0,
                           S1x_1,
                           S1x_2,
                           S1x_3,
                           ix_min,
                           i1(p),
                           dx1(p),
                           i1_prev(p),
                           dx1_prev(p));

        if constexpr (D == Dim::_1D) {
          // ToDo
        } else if constexpr (D == Dim::_2D) {

          /*
            y - direction
          */

          // shape function at previous timestep
          real_t   S0y_0, S0y_1, S0y_2, S0y_3;
          // shape function at current timestep
          real_t   S1y_0, S1y_1, S1y_2, S1y_3;
          // indices of the shape function
          ncells_t iy_min;
          // find indices and define shape function
          shape_function_2nd(S0y_0,
                             S0y_1,
                             S0y_2,
                             S0y_3,
                             S1y_0,
                             S1y_1,
                             S1y_2,
                             S1y_3,
                             iy_min,
                             i2(p),
                             dx2(p),
                             i2_prev(p),
                             dx2_prev(p));

          // Esirkepov 2001, Eq. 39
          /*
              x - component
          */
          // Calculate weight function - unrolled
          const auto Wx_0_0 = HALF * (S1x_0 - S0x_0) * (S0y_0 + S1y_0);
          const auto Wx_0_1 = HALF * (S1x_0 - S0x_0) * (S0y_1 + S1y_1);
          const auto Wx_0_2 = HALF * (S1x_0 - S0x_0) * (S0y_2 + S1y_2);
          const auto Wx_0_3 = HALF * (S1x_0 - S0x_0) * (S0y_3 + S1y_3);

          const auto Wx_1_0 = HALF * (S1x_1 - S0x_1) * (S0y_0 + S1y_0);
          const auto Wx_1_1 = HALF * (S1x_1 - S0x_1) * (S0y_1 + S1y_1);
          const auto Wx_1_2 = HALF * (S1x_1 - S0x_1) * (S0y_2 + S1y_2);
          const auto Wx_1_3 = HALF * (S1x_1 - S0x_1) * (S0y_3 + S1y_3);

          const auto Wx_2_0 = HALF * (S1x_2 - S0x_2) * (S0y_0 + S1y_0);
          const auto Wx_2_1 = HALF * (S1x_2 - S0x_2) * (S0y_1 + S1y_1);
          const auto Wx_2_2 = HALF * (S1x_2 - S0x_2) * (S0y_2 + S1y_2);
          const auto Wx_2_3 = HALF * (S1x_2 - S0x_2) * (S0y_3 + S1y_3);

          const auto Wx_3_0 = HALF * (S1x_3 - S0x_3) * (S0y_0 + S1y_0);
          const auto Wx_3_1 = HALF * (S1x_3 - S0x_3) * (S0y_1 + S1y_1);
          const auto Wx_3_2 = HALF * (S1x_3 - S0x_3) * (S0y_2 + S1y_2);
          const auto Wx_3_3 = HALF * (S1x_3 - S0x_3) * (S0y_3 + S1y_3);

          // Unrolled calculations for Wy
          const auto Wy_0_0 = HALF * (S1x_0 + S0x_0) * (S0y_0 - S1y_0);
          const auto Wy_0_1 = HALF * (S1x_0 + S0x_0) * (S0y_1 - S1y_1);
          const auto Wy_0_2 = HALF * (S1x_0 + S0x_0) * (S0y_2 - S1y_2);
          const auto Wy_0_3 = HALF * (S1x_0 + S0x_0) * (S0y_3 - S1y_3);

          const auto Wy_1_0 = HALF * (S1x_1 + S0x_1) * (S0y_0 - S1y_0);
          const auto Wy_1_1 = HALF * (S1x_1 + S0x_1) * (S0y_1 - S1y_1);
          const auto Wy_1_2 = HALF * (S1x_1 + S0x_1) * (S0y_2 - S1y_2);
          const auto Wy_1_3 = HALF * (S1x_1 + S0x_1) * (S0y_3 - S1y_3);

          const auto Wy_2_0 = HALF * (S1x_2 + S0x_2) * (S0y_0 - S1y_0);
          const auto Wy_2_1 = HALF * (S1x_2 + S0x_2) * (S0y_1 - S1y_1);
          const auto Wy_2_2 = HALF * (S1x_2 + S0x_2) * (S0y_2 - S1y_2);
          const auto Wy_2_3 = HALF * (S1x_2 + S0x_2) * (S0y_3 - S1y_3);

          const auto Wy_3_0 = HALF * (S1x_3 + S0x_3) * (S0y_0 - S1y_0);
          const auto Wy_3_1 = HALF * (S1x_3 + S0x_3) * (S0y_1 - S1y_1);
          const auto Wy_3_2 = HALF * (S1x_3 + S0x_3) * (S0y_2 - S1y_2);
          const auto Wy_3_3 = HALF * (S1x_3 + S0x_3) * (S0y_3 - S1y_3);

          // Unrolled calculations for Wz
          const auto Wz_0_0 = THIRD * (S1y_0 * (HALF * S0x_0 + S1x_0) +
                                       S0y_0 * (HALF * S1x_0 + S0x_0));
          const auto Wz_0_1 = THIRD * (S1y_1 * (HALF * S0x_0 + S1x_0) +
                                       S0y_1 * (HALF * S1x_0 + S0x_0));
          const auto Wz_0_2 = THIRD * (S1y_2 * (HALF * S0x_0 + S1x_0) +
                                       S0y_2 * (HALF * S1x_0 + S0x_0));
          const auto Wz_0_3 = THIRD * (S1y_3 * (HALF * S0x_0 + S1x_0) +
                                       S0y_3 * (HALF * S1x_0 + S0x_0));

          const auto Wz_1_0 = THIRD * (S1y_0 * (HALF * S0x_1 + S1x_1) +
                                       S0y_0 * (HALF * S1x_1 + S0x_1));
          const auto Wz_1_1 = THIRD * (S1y_1 * (HALF * S0x_1 + S1x_1) +
                                       S0y_1 * (HALF * S1x_1 + S0x_1));
          const auto Wz_1_2 = THIRD * (S1y_2 * (HALF * S0x_1 + S1x_1) +
                                       S0y_2 * (HALF * S1x_1 + S0x_1));
          const auto Wz_1_3 = THIRD * (S1y_3 * (HALF * S0x_1 + S1x_1) +
                                       S0y_3 * (HALF * S1x_1 + S0x_1));

          const auto Wz_2_0 = THIRD * (S1y_0 * (HALF * S0x_2 + S1x_2) +
                                       S0y_0 * (HALF * S1x_2 + S0x_2));
          const auto Wz_2_1 = THIRD * (S1y_1 * (HALF * S0x_2 + S1x_2) +
                                       S0y_1 * (HALF * S1x_2 + S0x_2));
          const auto Wz_2_2 = THIRD * (S1y_2 * (HALF * S0x_2 + S1x_2) +
                                       S0y_2 * (HALF * S1x_2 + S0x_2));
          const auto Wz_2_3 = THIRD * (S1y_3 * (HALF * S0x_2 + S1x_2) +
                                       S0y_3 * (HALF * S1x_2 + S0x_2));

          const auto Wz_3_0 = THIRD * (S1y_0 * (HALF * S0x_3 + S1x_3) +
                                       S0y_0 * (HALF * S1x_3 + S0x_3));
          const auto Wz_3_1 = THIRD * (S1y_1 * (HALF * S0x_3 + S1x_3) +
                                       S0y_1 * (HALF * S1x_3 + S0x_3));
          const auto Wz_3_2 = THIRD * (S1y_2 * (HALF * S0x_3 + S1x_3) +
                                       S0y_2 * (HALF * S1x_3 + S0x_3));
          const auto Wz_3_3 = THIRD * (S1y_3 * (HALF * S0x_3 + S1x_3) +
                                       S0y_3 * (HALF * S1x_3 + S0x_3));

          const auto delta_x = static_cast<real_t>(i1(p) == i1_prev(p)) *
                                 static_cast<real_t>(dx1(p) - dx1_prev(p)) +
                               static_cast<real_t>(i1(p) == i1_prev(p) + 1) *
                                 static_cast<real_t>(dx1(p) + (1 - dx1_prev(p))) +
                               static_cast<real_t>(i1(p) == i1_prev(p) - 1) *
                                 static_cast<real_t>((1 - dx1(p)) + dx1_prev(p));

          const auto delta_y = static_cast<real_t>(i2(p) == i2_prev(p)) *
                                 static_cast<real_t>(dx2(p) - dx2_prev(p)) +
                               static_cast<real_t>(i2(p) == i2_prev(p) + 1) *
                                 static_cast<real_t>(dx2(p) + (1 - dx2_prev(p))) +
                               static_cast<real_t>(i2(p) == i2_prev(p) - 1) *
                                 static_cast<real_t>((1 - dx2(p)) + dx2_prev(p));

          const real_t Qdxdt = -coeff * inv_dt * delta_x;
          const real_t Qdydt = -coeff * inv_dt * delta_y;
          const real_t QVz   = -coeff * vp[2];

          // Esirkepov - Eq. 32
          // x-component
          const auto jx_local_0_0 = -Qdxdt * Wx_0_0;
          const auto jx_local_1_0 = jx_local_0_0 - Qdxdt * Wx_1_0;
          const auto jx_local_2_0 = jx_local_1_0 - Qdxdt * Wx_2_0;
          const auto jx_local_3_0 = jx_local_2_0 - Qdxdt * Wx_3_0;

          const auto jx_local_0_1 = -Qdxdt * Wx_0_1;
          const auto jx_local_1_1 = jx_local_0_1 - Qdxdt * Wx_1_1;
          const auto jx_local_2_1 = jx_local_1_1 - Qdxdt * Wx_2_1;
          const auto jx_local_3_1 = jx_local_2_1 - Qdxdt * Wx_3_1;

          const auto jx_local_0_2 = -Qdxdt * Wx_0_2;
          const auto jx_local_1_2 = jx_local_0_2 - Qdxdt * Wx_1_2;
          const auto jx_local_2_2 = jx_local_1_2 - Qdxdt * Wx_2_2;
          const auto jx_local_3_2 = jx_local_2_2 - Qdxdt * Wx_3_2;

          const auto jx_local_0_3 = -Qdxdt * Wx_0_3;
          const auto jx_local_1_3 = jx_local_0_3 - Qdxdt * Wx_1_3;
          const auto jx_local_2_3 = jx_local_1_3 - Qdxdt * Wx_2_3;
          const auto jx_local_3_3 = jx_local_2_3 - Qdxdt * Wx_3_3;

          // y-component
          const auto jy_local_0_0 = -Qdydt * Wy_0_0;
          const auto jy_local_1_0 = jy_local_0_0 - Qdydt * Wy_1_0;
          const auto jy_local_2_0 = jy_local_1_0 - Qdydt * Wy_2_0;
          const auto jy_local_3_0 = jy_local_2_0 - Qdydt * Wy_3_0;

          const auto jy_local_0_1 = -Qdydt * Wy_0_1;
          const auto jy_local_1_1 = jy_local_0_1 - Qdydt * Wy_1_1;
          const auto jy_local_2_1 = jy_local_1_1 - Qdydt * Wy_2_1;
          const auto jy_local_3_1 = jy_local_2_1 - Qdydt * Wy_3_1;

          const auto jy_local_0_2 = -Qdydt * Wy_0_2;
          const auto jy_local_1_2 = jy_local_0_2 - Qdydt * Wy_1_2;
          const auto jy_local_2_2 = jy_local_1_2 - Qdydt * Wy_2_2;
          const auto jy_local_3_2 = jy_local_2_2 - Qdydt * Wy_3_2;

          const auto jy_local_0_3 = -Qdydt * Wy_0_3;
          const auto jy_local_1_3 = jy_local_0_3 - Qdydt * Wy_1_3;
          const auto jy_local_2_3 = jy_local_1_3 - Qdydt * Wy_2_3;
          const auto jy_local_3_3 = jy_local_2_3 - Qdydt * Wy_3_3;

          /*
            Current update
          */
          auto J_acc = J.access();

          /*
              x - component
          */
          J_acc(ix_min, iy_min, cur::jx1)     += jx_local_0_0;
          J_acc(ix_min, iy_min + 1, cur::jx1) += jx_local_0_1;
          J_acc(ix_min, iy_min + 2, cur::jx1) += jx_local_0_2;
          J_acc(ix_min, iy_min + 3, cur::jx1) += jx_local_0_3;
          
          J_acc(ix_min + 1, iy_min, cur::jx1)     += jx_local_1_0;
          J_acc(ix_min + 1, iy_min + 1, cur::jx1) += jx_local_1_1;
          J_acc(ix_min + 1, iy_min + 2, cur::jx1) += jx_local_1_2;
          J_acc(ix_min + 1, iy_min + 3, cur::jx1) += jx_local_1_3;
          
          J_acc(ix_min + 2, iy_min, cur::jx1)     += jx_local_2_0;
          J_acc(ix_min + 2, iy_min + 1, cur::jx1) += jx_local_2_1;
          J_acc(ix_min + 2, iy_min + 2, cur::jx1) += jx_local_2_2;
          J_acc(ix_min + 2, iy_min + 3, cur::jx1) += jx_local_2_3;
          
          J_acc(ix_min + 3, iy_min, cur::jx1)     += jx_local_3_0;
          J_acc(ix_min + 3, iy_min + 1, cur::jx1) += jx_local_3_1;
          J_acc(ix_min + 3, iy_min + 2, cur::jx1) += jx_local_3_2;
          J_acc(ix_min + 3, iy_min + 3, cur::jx1) += jx_local_3_3;

          /*
              y - component
          */
          J_acc(ix_min, iy_min, cur::jx2)     += jy_local_0_0;
          J_acc(ix_min, iy_min + 1, cur::jx2) += jy_local_0_1;
          J_acc(ix_min, iy_min + 2, cur::jx2) += jy_local_0_2;
          J_acc(ix_min, iy_min + 3, cur::jx2) += jy_local_0_3;

          J_acc(ix_min + 1, iy_min, cur::jx2)     += jy_local_1_0;
          J_acc(ix_min + 1, iy_min + 1, cur::jx2) += jy_local_1_1;
          J_acc(ix_min + 1, iy_min + 2, cur::jx2) += jy_local_1_2;
          J_acc(ix_min + 1, iy_min + 3, cur::jx2) += jy_local_1_3;

          J_acc(ix_min + 2, iy_min, cur::jx2)     += jy_local_2_0;
          J_acc(ix_min + 2, iy_min + 1, cur::jx2) += jy_local_2_1;
          J_acc(ix_min + 2, iy_min + 2, cur::jx2) += jy_local_2_2;
          J_acc(ix_min + 2, iy_min + 3, cur::jx2) += jy_local_2_3;

          J_acc(ix_min + 3, iy_min, cur::jx2)     += jy_local_3_0;
          J_acc(ix_min + 3, iy_min + 1, cur::jx2) += jy_local_3_1;
          J_acc(ix_min + 3, iy_min + 2, cur::jx2) += jy_local_3_2;
          J_acc(ix_min + 3, iy_min + 3, cur::jx2) += jy_local_3_3;

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

        } else if constexpr (D == Dim::_3D) {
          // /*
          //   y - direction
          // */
          //
          // // shape function at previous timestep
          // real_t S0y_0, S0y_1, S0y_2, S0y_3;
          // // shape function at current timestep
          // real_t S1y_0, S1y_1, S1y_2, S1y_3;
          // // indices of the shape function
          // uint   iy_min;
          // // find indices and define shape function
          // shape_function_2nd(S0y_0,
          //                    S0y_1,
          //                    S0y_2,
          //                    S0y_3,
          //                    S1y_0,
          //                    S1y_1,
          //                    S1y_2,
          //                    S1y_3,
          //                    iy_min,
          //                    i2(p),
          //                    dx2(p),
          //                    i2_prev(p),
          //                    dx2_prev(p));
          //
          // /*
          //   z - direction
          // */
          //
          // // shape function at previous timestep
          // real_t S0z_0, S0z_1, S0z_2, S0z_3;
          // // shape function at current timestep
          // real_t S1z_0, S1z_1, S1z_2, S1z_3;
          // // indices of the shape function
          // uint   iz_min;
          // // find indices and define shape function
          // shape_function_2nd(S0z_0,
          //                    S0z_1,
          //                    S0z_2,
          //                    S0z_3,
          //                    S1z_0,
          //                    S1z_1,
          //                    S1z_2,
          //                    S1z_3,
          //                    iz_min,
          //                    i3(p),
          //                    dx3(p),
          //                    i3_prev(p),
          //                    dx3_prev(p));
          //
          // // Calculate weight function
          // // for (int i = 0; i < interp_order + 2; ++i) {
          // //   for (int j = 0; j < interp_order + 2; ++j) {
          // //     for (int k = 0; k < interp_order + 2; ++k) {
          // //       // Esirkepov 2001, Eq. 31
          // //       Wx[i][j][k] = THIRD * (S1x[i] - S0x[i]) *
          // //                     ((S0y[j] * S0z[k] + S1y[j] * S1z[k]) +
          // //                      HALF * (S0z[k] * S1y[j] + S0y[j] * S1z[k]));
          // //
          // //       Wy[i][j][k] = THIRD * (S1y[j] - S0y[j]) *
          // //                     (S0x[i] * S0z[k] + S1x[i] * S1z[k] +
          // //                      HALF * (S0z[k] * S1x[i] + S0x[i] * S1z[k]));
          // //
          // //       Wz[i][j][k] = THIRD * (S1z[k] - S0z[k]) *
          // //                     (S0x[i] * S0y[j] + S1x[i] * S1y[j] +
          // //                      HALF * (S0x[i] * S1y[j] + S0y[j] * S1x[i]));
          // //     }
          // //   }
          // // }
          // //
          // // Unrolled calculations for Wx, Wy, and Wz
          // const auto Wx_0_0_0 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_0 * S0z_0 + S1y_0 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_0 + S0y_0 * S1z_0));
          // const auto Wx_0_0_1 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_0 * S0z_1 + S1y_0 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_0 + S0y_0 * S1z_1));
          // const auto Wx_0_0_2 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_0 * S0z_2 + S1y_0 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_0 + S0y_0 * S1z_2));
          // const auto Wx_0_0_3 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_0 * S0z_3 + S1y_0 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_0 + S0y_0 * S1z_3));
          //
          // const auto Wx_0_1_0 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_1 * S0z_0 + S1y_1 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_1 + S0y_1 * S1z_0));
          // const auto Wx_0_1_1 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_1 * S0z_1 + S1y_1 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_1 + S0y_1 * S1z_1));
          // const auto Wx_0_1_2 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_1 * S0z_2 + S1y_1 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_1 + S0y_1 * S1z_2));
          // const auto Wx_0_1_3 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_1 * S0z_3 + S1y_1 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_1 + S0y_1 * S1z_3));
          //
          // const auto Wx_0_2_0 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_2 * S0z_0 + S1y_2 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_2 + S0y_2 * S1z_0));
          // const auto Wx_0_2_1 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_2 * S0z_1 + S1y_2 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_2 + S0y_2 * S1z_1));
          // const auto Wx_0_2_2 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_2 * S0z_2 + S1y_2 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_2 + S0y_2 * S1z_2));
          // const auto Wx_0_2_3 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_2 * S0z_3 + S1y_2 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_2 + S0y_2 * S1z_3));
          //
          // const auto Wx_0_3_0 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_3 * S0z_0 + S1y_3 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_3 + S0y_3 * S1z_0));
          // const auto Wx_0_3_1 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_3 * S0z_1 + S1y_3 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_3 + S0y_3 * S1z_1));
          // const auto Wx_0_3_2 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_3 * S0z_2 + S1y_3 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_3 + S0y_3 * S1z_2));
          // const auto Wx_0_3_3 = THIRD * (S1x_0 - S0x_0) *
          //                       ((S0y_3 * S0z_3 + S1y_3 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_3 + S0y_3 * S1z_3));
          //
          // const auto Wx_1_0_0 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_0 * S0z_0 + S1y_0 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_0 + S0y_0 * S1z_0));
          // const auto Wx_1_0_1 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_0 * S0z_1 + S1y_0 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_0 + S0y_0 * S1z_1));
          // const auto Wx_1_0_2 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_0 * S0z_2 + S1y_0 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_0 + S0y_0 * S1z_2));
          // const auto Wx_1_0_3 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_0 * S0z_3 + S1y_0 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_0 + S0y_0 * S1z_3));
          //
          // const auto Wx_1_1_0 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_1 * S0z_0 + S1y_1 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_1 + S0y_1 * S1z_0));
          // const auto Wx_1_1_1 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_1 * S0z_1 + S1y_1 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_1 + S0y_1 * S1z_1));
          // const auto Wx_1_1_2 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_1 * S0z_2 + S1y_1 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_1 + S0y_1 * S1z_2));
          // const auto Wx_1_1_3 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_1 * S0z_3 + S1y_1 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_1 + S0y_1 * S1z_3));
          //
          // const auto Wx_1_2_0 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_2 * S0z_0 + S1y_2 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_2 + S0y_2 * S1z_0));
          // const auto Wx_1_2_1 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_2 * S0z_1 + S1y_2 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_2 + S0y_2 * S1z_1));
          // const auto Wx_1_2_2 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_2 * S0z_2 + S1y_2 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_2 + S0y_2 * S1z_2));
          // const auto Wx_1_2_3 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_2 * S0z_3 + S1y_2 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_2 + S0y_2 * S1z_3));
          //
          // const auto Wx_1_3_0 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_3 * S0z_0 + S1y_3 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_3 + S0y_3 * S1z_0));
          // const auto Wx_1_3_1 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_3 * S0z_1 + S1y_3 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_3 + S0y_3 * S1z_1));
          // const auto Wx_1_3_2 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_3 * S0z_2 + S1y_3 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_3 + S0y_3 * S1z_2));
          // const auto Wx_1_3_3 = THIRD * (S1x_1 - S0x_1) *
          //                       ((S0y_3 * S0z_3 + S1y_3 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_3 + S0y_3 * S1z_3));
          //
          // const auto Wx_2_0_0 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_0 * S0z_0 + S1y_0 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_0 + S0y_0 * S1z_0));
          // const auto Wx_2_0_1 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_0 * S0z_1 + S1y_0 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_0 + S0y_0 * S1z_1));
          // const auto Wx_2_0_2 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_0 * S0z_2 + S1y_0 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_0 + S0y_0 * S1z_2));
          // const auto Wx_2_0_3 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_0 * S0z_3 + S1y_0 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_0 + S0y_0 * S1z_3));
          //
          // const auto Wx_2_1_0 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_1 * S0z_0 + S1y_1 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_1 + S0y_1 * S1z_0));
          // const auto Wx_2_1_1 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_1 * S0z_1 + S1y_1 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_1 + S0y_1 * S1z_1));
          // const auto Wx_2_1_2 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_1 * S0z_2 + S1y_1 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_1 + S0y_1 * S1z_2));
          // const auto Wx_2_1_3 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_1 * S0z_3 + S1y_1 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_1 + S0y_1 * S1z_3));
          //
          // const auto Wx_2_2_0 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_2 * S0z_0 + S1y_2 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_2 + S0y_2 * S1z_0));
          // const auto Wx_2_2_1 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_2 * S0z_1 + S1y_2 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_2 + S0y_2 * S1z_1));
          // const auto Wx_2_2_2 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_2 * S0z_2 + S1y_2 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_2 + S0y_2 * S1z_2));
          // const auto Wx_2_2_3 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_2 * S0z_3 + S1y_2 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_2 + S0y_2 * S1z_3));
          //
          // const auto Wx_2_3_0 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_3 * S0z_0 + S1y_3 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_3 + S0y_3 * S1z_0));
          // const auto Wx_2_3_1 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_3 * S0z_1 + S1y_3 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_3 + S0y_3 * S1z_1));
          // const auto Wx_2_3_2 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_3 * S0z_2 + S1y_3 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_3 + S0y_3 * S1z_2));
          // const auto Wx_2_3_3 = THIRD * (S1x_2 - S0x_2) *
          //                       ((S0y_3 * S0z_3 + S1y_3 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_3 + S0y_3 * S1z_3));
          //
          // const auto Wx_3_0_0 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_0 * S0z_0 + S1y_0 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_0 + S0y_0 * S1z_0));
          // const auto Wx_3_0_1 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_0 * S0z_1 + S1y_0 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_0 + S0y_0 * S1z_1));
          // const auto Wx_3_0_2 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_0 * S0z_2 + S1y_0 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_0 + S0y_0 * S1z_2));
          // const auto Wx_3_0_3 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_0 * S0z_3 + S1y_0 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_0 + S0y_0 * S1z_3));
          //
          // const auto Wx_3_1_0 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_1 * S0z_0 + S1y_1 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_1 + S0y_1 * S1z_0));
          // const auto Wx_3_1_1 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_1 * S0z_1 + S1y_1 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_1 + S0y_1 * S1z_1));
          // const auto Wx_3_1_2 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_1 * S0z_2 + S1y_1 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_1 + S0y_1 * S1z_2));
          // const auto Wx_3_1_3 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_1 * S0z_3 + S1y_1 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_1 + S0y_1 * S1z_3));
          //
          // const auto Wx_3_2_0 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_2 * S0z_0 + S1y_2 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_2 + S0y_2 * S1z_0));
          // const auto Wx_3_2_1 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_2 * S0z_1 + S1y_2 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_2 + S0y_2 * S1z_1));
          // const auto Wx_3_2_2 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_2 * S0z_2 + S1y_2 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_2 + S0y_2 * S1z_2));
          // const auto Wx_3_2_3 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_2 * S0z_3 + S1y_2 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_2 + S0y_2 * S1z_3));
          //
          // const auto Wx_3_3_0 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_3 * S0z_0 + S1y_3 * S1z_0) +
          //                        HALF * (S0z_0 * S1y_3 + S0y_3 * S1z_0));
          // const auto Wx_3_3_1 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_3 * S0z_1 + S1y_3 * S1z_1) +
          //                        HALF * (S0z_1 * S1y_3 + S0y_3 * S1z_1));
          // const auto Wx_3_3_2 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_3 * S0z_2 + S1y_3 * S1z_2) +
          //                        HALF * (S0z_2 * S1y_3 + S0y_3 * S1z_2));
          // const auto Wx_3_3_3 = THIRD * (S1x_3 - S0x_3) *
          //                       ((S0y_3 * S0z_3 + S1y_3 * S1z_3) +
          //                        HALF * (S0z_3 * S1y_3 + S0y_3 * S1z_3));
          //
          // const real_t Qdxdt = coeff * inv_dt * dxp_r_1;
          //
          // J_acc(ix_min, iy_min, iz_min, cur::jx1)         += Qdxdt * Wx_0_0_0;
          // J_acc(ix_min, iy_min, iz_min + 1, cur::jx1)     += Qdxdt * Wx_0_0_1;
          // J_acc(ix_min, iy_min, iz_min + 2, cur::jx1)     += Qdxdt * Wx_0_0_2;
          // J_acc(ix_min, iy_min, iz_min + 3, cur::jx1)     += Qdxdt * Wx_0_0_3;
          // //
          // J_acc(ix_min, iy_min + 1, iz_min, cur::jx1)     += Qdxdt * Wx_0_1_0;
          // J_acc(ix_min, iy_min + 1, iz_min + 1, cur::jx1) += Qdxdt * Wx_0_1_1;
          // J_acc(ix_min, iy_min + 1, iz_min + 2, cur::jx1) += Qdxdt * Wx_0_1_2;
          // J_acc(ix_min, iy_min + 1, iz_min + 3, cur::jx1) += Qdxdt * Wx_0_1_3;
          // //
          // J_acc(ix_min, iy_min + 2, iz_min, cur::jx1)     += Qdxdt * Wx_0_2_0;
          // J_acc(ix_min, iy_min + 2, iz_min + 1, cur::jx1) += Qdxdt * Wx_0_2_1;
          // J_acc(ix_min, iy_min + 2, iz_min + 2, cur::jx1) += Qdxdt * Wx_0_2_2;
          // J_acc(ix_min, iy_min + 2, iz_min + 3, cur::jx1) += Qdxdt * Wx_0_2_3;
          // //
          // J_acc(ix_min, iy_min + 3, iz_min, cur::jx1)     += Qdxdt * Wx_0_3_0;
          // J_acc(ix_min, iy_min + 3, iz_min + 1, cur::jx1) += Qdxdt * Wx_0_3_1;
          // J_acc(ix_min, iy_min + 3, iz_min + 2, cur::jx1) += Qdxdt * Wx_0_3_2;
          // J_acc(ix_min, iy_min + 3, iz_min + 3, cur::jx1) += Qdxdt * Wx_0_3_3;
          // //
          // //
          // J_acc(ix_min + 1, iy_min, iz_min, cur::jx1)     += Qdxdt * Wx_1_0_0;
          // J_acc(ix_min + 1, iy_min, iz_min + 1, cur::jx1) += Qdxdt * Wx_1_0_1;
          // J_acc(ix_min + 1, iy_min, iz_min + 2, cur::jx1) += Qdxdt * Wx_1_0_2;
          // J_acc(ix_min + 1, iy_min, iz_min + 3, cur::jx1) += Qdxdt * Wx_1_0_3;
          // //
          // J_acc(ix_min + 1, iy_min + 1, iz_min, cur::jx1) += Qdxdt * Wx_1_1_0;
          // J_acc(ix_min + 1, iy_min + 1, iz_min + 1, cur::jx1) += Qdxdt * Wx_1_1_1;
          // J_acc(ix_min + 1, iy_min + 1, iz_min + 2, cur::jx1) += Qdxdt * Wx_1_1_2;
          // J_acc(ix_min + 1, iy_min + 1, iz_min + 3, cur::jx1) += Qdxdt * Wx_1_1_3;
          // //
          // J_acc(ix_min + 1, iy_min + 2, iz_min, cur::jx1) += Qdxdt * Wx_1_2_0;
          // J_acc(ix_min + 1, iy_min + 2, iz_min + 1, cur::jx1) += Qdxdt * Wx_1_2_1;
          // J_acc(ix_min + 1, iy_min + 2, iz_min + 2, cur::jx1) += Qdxdt * Wx_1_2_2;
          // J_acc(ix_min + 1, iy_min + 2, iz_min + 3, cur::jx1) += Qdxdt * Wx_1_2_3;
          // //
          // J_acc(ix_min + 1, iy_min + 3, iz_min, cur::jx1) += Qdxdt * Wx_1_3_0;
          // J_acc(ix_min + 1, iy_min + 3, iz_min + 1, cur::jx1) += Qdxdt * Wx_1_3_1;
          // J_acc(ix_min + 1, iy_min + 3, iz_min + 2, cur::jx1) += Qdxdt * Wx_1_3_2;
          // J_acc(ix_min + 1, iy_min + 3, iz_min + 3, cur::jx1) += Qdxdt * Wx_1_3_3;
          // //
          // //
          // J_acc(ix_min + 2, iy_min, iz_min, cur::jx1)     += Qdxdt * Wx_2_0_0;
          // J_acc(ix_min + 2, iy_min, iz_min + 1, cur::jx1) += Qdxdt * Wx_2_0_1;
          // J_acc(ix_min + 2, iy_min, iz_min + 2, cur::jx1) += Qdxdt * Wx_2_0_2;
          // J_acc(ix_min + 2, iy_min, iz_min + 3, cur::jx1) += Qdxdt * Wx_2_0_3;
          // //
          // J_acc(ix_min + 2, iy_min + 1, iz_min, cur::jx1) += Qdxdt * Wx_2_1_0;
          // J_acc(ix_min + 2, iy_min + 1, iz_min + 1, cur::jx1) += Qdxdt * Wx_2_1_1;
          // J_acc(ix_min + 2, iy_min + 1, iz_min + 2, cur::jx1) += Qdxdt * Wx_2_1_2;
          // J_acc(ix_min + 2, iy_min + 1, iz_min + 3, cur::jx1) += Qdxdt * Wx_2_1_3;
          // //
          // J_acc(ix_min + 2, iy_min + 2, iz_min, cur::jx1) += Qdxdt * Wx_2_2_0;
          // J_acc(ix_min + 2, iy_min + 2, iz_min + 1, cur::jx1) += Qdxdt * Wx_2_2_1;
          // J_acc(ix_min + 2, iy_min + 2, iz_min + 2, cur::jx1) += Qdxdt * Wx_2_2_2;
          // J_acc(ix_min + 2, iy_min + 2, iz_min + 3, cur::jx1) += Qdxdt * Wx_2_2_3;
          // //
          // J_acc(ix_min + 2, iy_min + 3, iz_min, cur::jx1) += Qdxdt * Wx_2_3_0;
          // J_acc(ix_min + 2, iy_min + 3, iz_min + 1, cur::jx1) += Qdxdt * Wx_2_3_1;
          // J_acc(ix_min + 2, iy_min + 3, iz_min + 2, cur::jx1) += Qdxdt * Wx_2_3_2;
          // J_acc(ix_min + 2, iy_min + 3, iz_min + 3, cur::jx1) += Qdxdt * Wx_2_3_3;
          // //
          // //
          // J_acc(ix_min + 3, iy_min, iz_min, cur::jx1)     += Qdxdt * Wx_3_0_0;
          // J_acc(ix_min + 3, iy_min, iz_min + 1, cur::jx1) += Qdxdt * Wx_3_0_1;
          // J_acc(ix_min + 3, iy_min, iz_min + 2, cur::jx1) += Qdxdt * Wx_3_0_2;
          // J_acc(ix_min + 3, iy_min, iz_min + 3, cur::jx1) += Qdxdt * Wx_3_0_3;
          // //
          // J_acc(ix_min + 3, iy_min + 1, iz_min, cur::jx1) += Qdxdt * Wx_3_1_0;
          // J_acc(ix_min + 3, iy_min + 1, iz_min + 1, cur::jx1) += Qdxdt * Wx_3_1_1;
          // J_acc(ix_min + 3, iy_min + 1, iz_min + 2, cur::jx1) += Qdxdt * Wx_3_1_2;
          // J_acc(ix_min + 3, iy_min + 1, iz_min + 3, cur::jx1) += Qdxdt * Wx_3_1_3;
          // //
          // J_acc(ix_min + 3, iy_min + 2, iz_min, cur::jx1) += Qdxdt * Wx_3_2_0;
          // J_acc(ix_min + 3, iy_min + 2, iz_min + 1, cur::jx1) += Qdxdt * Wx_3_2_1;
          // J_acc(ix_min + 3, iy_min + 2, iz_min + 2, cur::jx1) += Qdxdt * Wx_3_2_2;
          // J_acc(ix_min + 3, iy_min + 2, iz_min + 3, cur::jx1) += Qdxdt * Wx_3_2_3;
          // //
          // J_acc(ix_min + 3, iy_min + 3, iz_min, cur::jx1) += Qdxdt * Wx_3_3_0;
          // J_acc(ix_min + 3, iy_min + 3, iz_min + 1, cur::jx1) += Qdxdt * Wx_3_3_1;
          // J_acc(ix_min + 3, iy_min + 3, iz_min + 2, cur::jx1) += Qdxdt * Wx_3_3_2;
          // J_acc(ix_min + 3, iy_min + 3, iz_min + 3, cur::jx1) += Qdxdt * Wx_3_3_3;
          //
          // /*
          //   y-component
          // */
          // // i = 0
          // const auto Wy_0_0_0 = THIRD * (S1y_0 - S0y_0) *
          //                       (S0x_0 * S0z_0 + S1x_0 * S1z_0 +
          //                        HALF * (S0z_0 * S1x_0 + S0x_0 * S1z_0));
          // const auto Wy_0_0_1 = THIRD * (S1y_0 - S0y_0) *
          //                       (S0x_0 * S0z_1 + S1x_0 * S1z_1 +
          //                        HALF * (S0z_1 * S1x_0 + S0x_0 * S1z_1));
          // const auto Wy_0_0_2 = THIRD * (S1y_0 - S0y_0) *
          //                       (S0x_0 * S0z_2 + S1x_0 * S1z_2 +
          //                        HALF * (S0z_2 * S1x_0 + S0x_0 * S1z_2));
          // const auto Wy_0_0_3 = THIRD * (S1y_0 - S0y_0) *
          //                       (S0x_0 * S0z_3 + S1x_0 * S1z_3 +
          //                        HALF * (S0z_3 * S1x_0 + S0x_0 * S1z_3));
          //
          // const auto Wy_0_1_0 = THIRD * (S1y_1 - S0y_1) *
          //                       (S0x_0 * S0z_0 + S1x_0 * S1z_0 +
          //                        HALF * (S0z_0 * S1x_0 + S0x_0 * S1z_0));
          // const auto Wy_0_1_1 = THIRD * (S1y_1 - S0y_1) *
          //                       (S0x_0 * S0z_1 + S1x_0 * S1z_1 +
          //                        HALF * (S0z_1 * S1x_0 + S0x_0 * S1z_1));
          // const auto Wy_0_1_2 = THIRD * (S1y_1 - S0y_1) *
          //                       (S0x_0 * S0z_2 + S1x_0 * S1z_2 +
          //                        HALF * (S0z_2 * S1x_0 + S0x_0 * S1z_2));
          // const auto Wy_0_1_3 = THIRD * (S1y_1 - S0y_1) *
          //                       (S0x_0 * S0z_3 + S1x_0 * S1z_3 +
          //                        HALF * (S0z_3 * S1x_0 + S0x_0 * S1z_3));
          //
          // const auto Wy_0_2_0 = THIRD * (S1y_2 - S0y_2) *
          //                       (S0x_0 * S0z_0 + S1x_0 * S1z_0 +
          //                        HALF * (S0z_0 * S1x_0 + S0x_0 * S1z_0));
          // const auto Wy_0_2_1 = THIRD * (S1y_2 - S0y_2) *
          //                       (S0x_0 * S0z_1 + S1x_0 * S1z_1 +
          //                        HALF * (S0z_1 * S1x_0 + S0x_0 * S1z_1));
          // const auto Wy_0_2_2 = THIRD * (S1y_2 - S0y_2) *
          //                       (S0x_0 * S0z_2 + S1x_0 * S1z_2 +
          //                        HALF * (S0z_2 * S1x_0 + S0x_0 * S1z_2));
          // const auto Wy_0_2_3 = THIRD * (S1y_2 - S0y_2) *
          //                       (S0x_0 * S0z_3 + S1x_0 * S1z_3 +
          //                        HALF * (S0z_3 * S1x_0 + S0x_0 * S1z_3));
          //
          // const auto Wy_0_3_0 = THIRD * (S1y_3 - S0y_3) *
          //                       (S0x_0 * S0z_0 + S1x_0 * S1z_0 +
          //                        HALF * (S0z_0 * S1x_0 + S0x_0 * S1z_0));
          // const auto Wy_0_3_1 = THIRD * (S1y_3 - S0y_3) *
          //                       (S0x_0 * S0z_1 + S1x_0 * S1z_1 +
          //                        HALF * (S0z_1 * S1x_0 + S0x_0 * S1z_1));
          // const auto Wy_0_3_2 = THIRD * (S1y_3 - S0y_3) *
          //                       (S0x_0 * S0z_2 + S1x_0 * S1z_2 +
          //                        HALF * (S0z_2 * S1x_0 + S0x_0 * S1z_2));
          // const auto Wy_0_3_3 = THIRD * (S1y_3 - S0y_3) *
          //                       (S0x_0 * S0z_3 + S1x_0 * S1z_3 +
          //                        HALF * (S0z_3 * S1x_0 + S0x_0 * S1z_3));
          //
          // const auto Wy_1_0_0 = THIRD * (S1y_0 - S0y_0) *
          //                       (S0x_1 * S0z_0 + S1x_1 * S1z_0 +
          //                        HALF * (S0z_0 * S1x_1 + S0x_1 * S1z_0));
          // const auto Wy_1_0_1 = THIRD * (S1y_0 - S0y_0) *
          //                       (S0x_1 * S0z_1 + S1x_1 * S1z_1 +
          //                        HALF * (S0z_1 * S1x_1 + S0x_1 * S1z_1));
          // const auto Wy_1_0_2 = THIRD * (S1y_0 - S0y_0) *
          //                       (S0x_1 * S0z_2 + S1x_1 * S1z_2 +
          //                        HALF * (S0z_2 * S1x_1 + S0x_1 * S1z_2));
          // const auto Wy_1_0_3 = THIRD * (S1y_0 - S0y_0) *
          //                       (S0x_1 * S0z_3 + S1x_1 * S1z_3 +
          //                        HALF * (S0z_3 * S1x_1 + S0x_1 * S1z_3));
          //
          // const auto Wy_1_1_0 = THIRD * (S1y_1 - S0y_1) *
          //                       (S0x_1 * S0z_0 + S1x_1 * S1z_0 +
          //                        HALF * (S0z_0 * S1x_1 + S0x_1 * S1z_0));
          // const auto Wy_1_1_1 = THIRD * (S1y_1 - S0y_1) *
          //                       (S0x_1 * S0z_1 + S1x_1 * S1z_1 +
          //                        HALF * (S0z_1 * S1x_1 + S0x_1 * S1z_1));
          // const auto Wy_1_1_2 = THIRD * (S1y_1 - S0y_1) *
          //                       (S0x_1 * S0z_2 + S1x_1 * S1z_2 +
          //                        HALF * (S0z_2 * S1x_1 + S0x_1 * S1z_2));
          // const auto Wy_1_1_3 = THIRD * (S1y_1 - S0y_1) *
          //                       (S0x_1 * S0z_3 + S1x_1 * S1z_3 +
          //                        HALF * (S0z_3 * S1x_1 + S0x_1 * S1z_3));
          //
          // const auto Wy_1_2_0 = THIRD * (S1y_2 - S0y_2) *
          //                       (S0x_1 * S0z_0 + S1x_1 * S1z_0 +
          //                        HALF * (S0z_0 * S1x_1 + S0x_1 * S1z_0));
          // const auto Wy_1_2_1 = THIRD * (S1y_2 - S0y_2) *
          //                       (S0x_1 * S0z_1 + S1x_1 * S1z_1 +
          //                        HALF * (S0z_1 * S1x_1 + S0x_1 * S1z_1));
          // const auto Wy_1_2_2 = THIRD * (S1y_2 - S0y_2) *
          //                       (S0x_1 * S0z_2 + S1x_1 * S1z_2 +
          //                        HALF * (S0z_2 * S1x_1 + S0x_1 * S1z_2));
          // const auto Wy_1_2_3 = THIRD * (S1y_2 - S0y_2) *
          //                       (S0x_1 * S0z_3 + S1x_1 * S1z_3 +
          //                        HALF * (S0z_3 * S1x_1 + S0x_1 * S1z_3));
          //
          // const auto Wy_1_3_0 = THIRD * (S1y_3 - S0y_3) *
          //                       (S0x_1 * S0z_0 + S1x_1 * S1z_0 +
          //                        HALF * (S0z_0 * S1x_1 + S0x_1 * S1z_0));
          // const auto Wy_1_3_1 = THIRD * (S1y_3 - S0y_3) *
          //                       (S0x_1 * S0z_1 + S1x_1 * S1z_1 +
          //                        HALF * (S0z_1 * S1x_1 + S0x_1 * S1z_1));
          // const auto Wy_1_3_2 = THIRD * (S1y_3 - S0y_3) *
          //                       (S0x_1 * S0z_2 + S1x_1 * S1z_2 +
          //                        HALF * (S0z_2 * S1x_1 + S0x_1 * S1z_2));
          // const auto Wy_1_3_3 = THIRD * (S1y_3 - S0y_3) *
          //                       (S0x_1 * S0z_3 + S1x_1 * S1z_3 +
          //                        HALF * (S0z_3 * S1x_1 + S0x_1 * S1z_3));
          //
          // const auto Wy_2_0_0 = THIRD * (S1y_0 - S0y_0) *
          //                       (S0x_2 * S0z_0 + S1x_2 * S1z_0 +
          //                        HALF * (S0z_0 * S1x_2 + S0x_2 * S1z_0));
          // const auto Wy_2_0_1 = THIRD * (S1y_0 - S0y_0) *
          //                       (S0x_2 * S0z_1 + S1x_2 * S1z_1 +
          //                        HALF * (S0z_1 * S1x_2 + S0x_2 * S1z_1));
          // const auto Wy_2_0_2 = THIRD * (S1y_0 - S0y_0) *
          //                       (S0x_2 * S0z_2 + S1x_2 * S1z_2 +
          //                        HALF * (S0z_2 * S1x_2 + S0x_2 * S1z_2));
          // const auto Wy_2_0_3 = THIRD * (S1y_0 - S0y_0) *
          //                       (S0x_2 * S0z_3 + S1x_2 * S1z_3 +
          //                        HALF * (S0z_3 * S1x_2 + S0x_2 * S1z_3));
          //
          // const auto Wy_2_1_0 = THIRD * (S1y_1 - S0y_1) *
          //                       (S0x_2 * S0z_0 + S1x_2 * S1z_0 +
          //                        HALF * (S0z_0 * S1x_2 + S0x_2 * S1z_0));
          // const auto Wy_2_1_1 = THIRD * (S1y_1 - S0y_1) *
          //                       (S0x_2 * S0z_1 + S1x_2 * S1z_1 +
          //                        HALF * (S0z_1 * S1x_2 + S0x_2 * S1z_1));
          // const auto Wy_2_1_2 = THIRD * (S1y_1 - S0y_1) *
          //                       (S0x_2 * S0z_2 + S1x_2 * S1z_2 +
          //                        HALF * (S0z_2 * S1x_2 + S0x_2 * S1z_2));
          // const auto Wy_2_1_3 = THIRD * (S1y_1 - S0y_1) *
          //                       (S0x_2 * S0z_3 + S1x_2 * S1z_3 +
          //                        HALF * (S0z_3 * S1x_2 + S0x_2 * S1z_3));
          //
          // const auto Wy_2_2_0 = THIRD * (S1y_2 - S0y_2) *
          //                       (S0x_2 * S0z_0 + S1x_2 * S1z_0 +
          //                        HALF * (S0z_0 * S1x_2 + S0x_2 * S1z_0));
        