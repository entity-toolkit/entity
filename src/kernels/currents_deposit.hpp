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
                           const real_t&                  charge,
                           const real_t&                  dt)
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
      // _f = final, _i = initial
      tuple_t<int, D> Ip_f, Ip_i;
      coord_t<D>      xp_f, xp_i, xp_r;
      vec_t<Dim::_3D> vp { ZERO };

      // get [i, di]_init and [i, di]_final (per dimension)
      getDepositInterval(p, Ip_f, Ip_i, xp_f, xp_i, xp_r);
      // recover particle velocity to deposit in unsimulated direction
      getPrtl3Vel(p, vp);
      const real_t coeff { weight(p) * charge };
      depositCurrentsFromParticle(coeff, vp, Ip_f, Ip_i, xp_f, xp_i, xp_r);
    }

    /**
     * @brief Deposit currents from a single particle.
     * @param[in] coeff Particle weight x charge.
     * @param[in] vp Particle 3-velocity.
     * @param[in] Ip_f Final position of the particle (cell index).
     * @param[in] Ip_i Initial position of the particle (cell index).
     * @param[in] xp_f Final position.
     * @param[in] xp_i Previous step position.
     * @param[in] xp_r Intermediate point used in zig-zag deposit.
     */
    Inline auto depositCurrentsFromParticle(const real_t&          coeff,
                                            const vec_t<Dim::_3D>& vp,
                                            const tuple_t<int, D>& Ip_f,
                                            const tuple_t<int, D>& Ip_i,
                                            const coord_t<D>&      xp_f,
                                            const coord_t<D>&      xp_i,
                                            const coord_t<D>& xp_r) const -> void {
      const real_t Wx1_1 { HALF * (xp_i[0] + xp_r[0]) -
                           static_cast<real_t>(Ip_i[0]) };
      const real_t Wx1_2 { HALF * (xp_f[0] + xp_r[0]) -
                           static_cast<real_t>(Ip_f[0]) };
      const real_t Fx1_1 { (xp_r[0] - xp_i[0]) * coeff * inv_dt };
      const real_t Fx1_2 { (xp_f[0] - xp_r[0]) * coeff * inv_dt };

      auto J_acc = J.access();

      if constexpr (D == Dim::_1D) {
        const real_t Fx2_1 { HALF * vp[1] * coeff };
        const real_t Fx2_2 { HALF * vp[1] * coeff };

        const real_t Fx3_1 { HALF * vp[2] * coeff };
        const real_t Fx3_2 { HALF * vp[2] * coeff };

        J_acc(Ip_i[0] + N_GHOSTS, cur::jx1) += Fx1_1;
        J_acc(Ip_f[0] + N_GHOSTS, cur::jx1) += Fx1_2;

        J_acc(Ip_i[0] + N_GHOSTS, cur::jx2)     += Fx2_1 * (ONE - Wx1_1);
        J_acc(Ip_i[0] + N_GHOSTS + 1, cur::jx2) += Fx2_1 * Wx1_1;
        J_acc(Ip_f[0] + N_GHOSTS, cur::jx2)     += Fx2_2 * (ONE - Wx1_2);
        J_acc(Ip_f[0] + N_GHOSTS + 1, cur::jx2) += Fx2_2 * Wx1_2;

        J_acc(Ip_i[0] + N_GHOSTS, cur::jx3)     += Fx3_1 * (ONE - Wx1_1);
        J_acc(Ip_i[0] + N_GHOSTS + 1, cur::jx3) += Fx3_1 * Wx1_1;
        J_acc(Ip_f[0] + N_GHOSTS, cur::jx3)     += Fx3_2 * (ONE - Wx1_2);
        J_acc(Ip_f[0] + N_GHOSTS + 1, cur::jx3) += Fx3_2 * Wx1_2;
      } else if constexpr (D == Dim::_2D || D == Dim::_3D) {
        const real_t Wx2_1 { HALF * (xp_i[1] + xp_r[1]) -
                             static_cast<real_t>(Ip_i[1]) };
        const real_t Wx2_2 { HALF * (xp_f[1] + xp_r[1]) -
                             static_cast<real_t>(Ip_f[1]) };
        const real_t Fx2_1 { (xp_r[1] - xp_i[1]) * coeff * inv_dt };
        const real_t Fx2_2 { (xp_f[1] - xp_r[1]) * coeff * inv_dt };

        if constexpr (D == Dim::_2D) {
          const real_t Fx3_1 { HALF * vp[2] * coeff };
          const real_t Fx3_2 { HALF * vp[2] * coeff };

          J_acc(Ip_i[0] + N_GHOSTS, Ip_i[1] + N_GHOSTS, cur::jx1) += Fx1_1 *
                                                                     (ONE - Wx2_1);
          J_acc(Ip_i[0] + N_GHOSTS, Ip_i[1] + N_GHOSTS + 1, cur::jx1) += Fx1_1 *
                                                                         Wx2_1;
          J_acc(Ip_f[0] + N_GHOSTS, Ip_f[1] + N_GHOSTS, cur::jx1) += Fx1_2 *
                                                                     (ONE - Wx2_2);
          J_acc(Ip_f[0] + N_GHOSTS, Ip_f[1] + N_GHOSTS + 1, cur::jx1) += Fx1_2 *
                                                                         Wx2_2;

          J_acc(Ip_i[0] + N_GHOSTS, Ip_i[1] + N_GHOSTS, cur::jx2) += Fx2_1 *
                                                                     (ONE - Wx1_1);
          J_acc(Ip_i[0] + N_GHOSTS + 1, Ip_i[1] + N_GHOSTS, cur::jx2) += Fx2_1 *
                                                                         Wx1_1;
          J_acc(Ip_f[0] + N_GHOSTS, Ip_f[1] + N_GHOSTS, cur::jx2) += Fx2_2 *
                                                                     (ONE - Wx1_2);
          J_acc(Ip_f[0] + N_GHOSTS + 1, Ip_f[1] + N_GHOSTS, cur::jx2) += Fx2_2 *
                                                                         Wx1_2;

          J_acc(Ip_i[0] + N_GHOSTS,
                Ip_i[1] + N_GHOSTS,
                cur::jx3) += Fx3_1 * (ONE - Wx1_1) * (ONE - Wx2_1);
          J_acc(Ip_i[0] + N_GHOSTS + 1,
                Ip_i[1] + N_GHOSTS,
                cur::jx3) += Fx3_1 * Wx1_2 * (ONE - Wx2_1);
          J_acc(Ip_i[0] + N_GHOSTS,
                Ip_i[1] + N_GHOSTS + 1,
                cur::jx3) += Fx3_1 * (ONE - Wx1_1) * Wx2_1;
          J_acc(Ip_i[0] + N_GHOSTS + 1,
                Ip_i[1] + N_GHOSTS + 1,
                cur::jx3) += Fx3_1 * Wx1_1 * Wx2_1;

          J_acc(Ip_f[0] + N_GHOSTS,
                Ip_f[1] + N_GHOSTS,
                cur::jx3) += Fx3_2 * (ONE - Wx1_2) * (ONE - Wx2_2);
          J_acc(Ip_f[0] + N_GHOSTS + 1,
                Ip_f[1] + N_GHOSTS,
                cur::jx3) += Fx3_2 * Wx1_2 * (ONE - Wx2_2);
          J_acc(Ip_f[0] + N_GHOSTS,
                Ip_f[1] + N_GHOSTS + 1,
                cur::jx3) += Fx3_2 * (ONE - Wx1_2) * Wx2_2;
          J_acc(Ip_f[0] + N_GHOSTS + 1,
                Ip_f[1] + N_GHOSTS + 1,
                cur::jx3) += Fx3_2 * Wx1_2 * Wx2_2;
        } else {
          const real_t Wx3_1 { HALF * (xp_i[2] + xp_r[2]) -
                               static_cast<real_t>(Ip_i[2]) };
          const real_t Wx3_2 { HALF * (xp_f[2] + xp_r[2]) -
                               static_cast<real_t>(Ip_f[2]) };
          const real_t Fx3_1 { (xp_r[2] - xp_i[2]) * coeff * inv_dt };
          const real_t Fx3_2 { (xp_f[2] - xp_r[2]) * coeff * inv_dt };

          J_acc(Ip_i[0] + N_GHOSTS,
                Ip_i[1] + N_GHOSTS,
                Ip_i[2] + N_GHOSTS,
                cur::jx1) += Fx1_1 * (ONE - Wx2_1) * (ONE - Wx3_1);
          J_acc(Ip_i[0] + N_GHOSTS,
                Ip_i[1] + N_GHOSTS + 1,
                Ip_i[2] + N_GHOSTS,
                cur::jx1) += Fx1_1 * Wx2_1 * (ONE - Wx3_1);
          J_acc(Ip_i[0] + N_GHOSTS,
                Ip_i[1] + N_GHOSTS,
                Ip_i[2] + N_GHOSTS + 1,
                cur::jx1) += Fx1_1 * (ONE - Wx2_1) * Wx3_1;
          J_acc(Ip_i[0] + N_GHOSTS,
                Ip_i[1] + N_GHOSTS + 1,
                Ip_i[2] + N_GHOSTS + 1,
                cur::jx1) += Fx1_1 * Wx2_1 * Wx3_1;

          J_acc(Ip_f[0] + N_GHOSTS,
                Ip_f[1] + N_GHOSTS,
                Ip_f[2] + N_GHOSTS,
                cur::jx1) += Fx1_2 * (ONE - Wx2_2) * (ONE - Wx3_2);
          J_acc(Ip_f[0] + N_GHOSTS,
                Ip_f[1] + N_GHOSTS + 1,
                Ip_f[2] + N_GHOSTS,
                cur::jx1) += Fx1_2 * Wx2_2 * (ONE - Wx3_2);
          J_acc(Ip_f[0] + N_GHOSTS,
                Ip_f[1] + N_GHOSTS,
                Ip_f[2] + N_GHOSTS + 1,
                cur::jx1) += Fx1_2 * (ONE - Wx2_2) * Wx3_2;
          J_acc(Ip_f[0] + N_GHOSTS,
                Ip_f[1] + N_GHOSTS + 1,
                Ip_f[2] + N_GHOSTS + 1,
                cur::jx1) += Fx1_2 * Wx2_2 * Wx3_2;

          J_acc(Ip_i[0] + N_GHOSTS,
                Ip_i[1] + N_GHOSTS,
                Ip_i[2] + N_GHOSTS,
                cur::jx2) += Fx2_1 * (ONE - Wx1_1) * (ONE - Wx3_1);
          J_acc(Ip_i[0] + N_GHOSTS + 1,
                Ip_i[1] + N_GHOSTS,
                Ip_i[2] + N_GHOSTS,
                cur::jx2) += Fx2_1 * Wx1_1 * (ONE - Wx3_1);
          J_acc(Ip_i[0] + N_GHOSTS,
                Ip_i[1] + N_GHOSTS,
                Ip_i[2] + N_GHOSTS + 1,
                cur::jx2) += Fx2_1 * (ONE - Wx1_1) * Wx3_1;
          J_acc(Ip_i[0] + N_GHOSTS + 1,
                Ip_i[1] + N_GHOSTS,
                Ip_i[2] + N_GHOSTS + 1,
                cur::jx2) += Fx2_1 * Wx1_1 * Wx3_1;

          J_acc(Ip_f[0] + N_GHOSTS,
                Ip_f[1] + N_GHOSTS,
                Ip_f[2] + N_GHOSTS,
                cur::jx2) += Fx2_2 * (ONE - Wx1_2) * (ONE - Wx3_2);
          J_acc(Ip_f[0] + N_GHOSTS + 1,
                Ip_f[1] + N_GHOSTS,
                Ip_f[2] + N_GHOSTS,
                cur::jx2) += Fx2_2 * Wx1_2 * (ONE - Wx3_2);
          J_acc(Ip_f[0] + N_GHOSTS,
                Ip_f[1] + N_GHOSTS,
                Ip_f[2] + N_GHOSTS + 1,
                cur::jx2) += Fx2_2 * (ONE - Wx1_2) * Wx3_2;
          J_acc(Ip_f[0] + N_GHOSTS + 1,
                Ip_f[1] + N_GHOSTS,
                Ip_f[2] + N_GHOSTS + 1,
                cur::jx2) += Fx2_2 * Wx1_2 * Wx3_2;

          J_acc(Ip_i[0] + N_GHOSTS,
                Ip_i[1] + N_GHOSTS,
                Ip_i[2] + N_GHOSTS,
                cur::jx3) += Fx3_1 * (ONE - Wx1_1) * (ONE - Wx2_1);
          J_acc(Ip_i[0] + N_GHOSTS + 1,
                Ip_i[1] + N_GHOSTS,
                Ip_i[2] + N_GHOSTS,
                cur::jx3) += Fx3_1 * Wx1_1 * (ONE - Wx2_1);
          J_acc(Ip_i[0] + N_GHOSTS,
                Ip_i[1] + N_GHOSTS + 1,
                Ip_i[2] + N_GHOSTS,
                cur::jx3) += Fx3_1 * (ONE - Wx1_1) * Wx2_1;
          J_acc(Ip_i[0] + N_GHOSTS + 1,
                Ip_i[1] + N_GHOSTS + 1,
                Ip_i[2] + N_GHOSTS,
                cur::jx3) += Fx3_1 * Wx1_1 * Wx2_1;

          J_acc(Ip_f[0] + N_GHOSTS,
                Ip_f[1] + N_GHOSTS,
                Ip_f[2] + N_GHOSTS,
                cur::jx3) += Fx3_2 * (ONE - Wx1_2) * (ONE - Wx2_2);
          J_acc(Ip_f[0] + N_GHOSTS + 1,
                Ip_f[1] + N_GHOSTS,
                Ip_f[2] + N_GHOSTS,
                cur::jx3) += Fx3_2 * Wx1_2 * (ONE - Wx2_2);
          J_acc(Ip_f[0] + N_GHOSTS,
                Ip_f[1] + N_GHOSTS + 1,
                Ip_f[2] + N_GHOSTS,
                cur::jx3) += Fx3_2 * (ONE - Wx1_2) * Wx2_2;
          J_acc(Ip_f[0] + N_GHOSTS + 1,
                Ip_f[1] + N_GHOSTS + 1,
                Ip_f[2] + N_GHOSTS,
                cur::jx3) += Fx3_2 * Wx1_2 * Wx2_2;
        }
      }
    }

    /**
     * @brief Get particle position in `coord_t` form.
     * @param[in] p Index of particle.
     * @param[out] Ip_f Final position of the particle (cell index).
     * @param[out] Ip_i Initial position of the particle (cell index).
     * @param[out] xp_f Final position.
     * @param[out] xp_i Previous step position.
     * @param[out] xp_r Intermediate point used in zig-zag deposit.
     */
    Inline auto getDepositInterval(index_t&         p,
                                   tuple_t<int, D>& Ip_f,
                                   tuple_t<int, D>& Ip_i,
                                   coord_t<D>&      xp_f,
                                   coord_t<D>&      xp_i,
                                   coord_t<D>&      xp_r) const -> void {
      Ip_f[0] = i1(p);
      Ip_i[0] = i1_prev(p);
      xp_f[0] = i_di_to_Xi(Ip_f[0], dx1(p));
      xp_i[0] = i_di_to_Xi(Ip_i[0], dx1_prev(p));
      if constexpr (D == Dim::_2D || D == Dim::_3D) {
        Ip_f[1] = i2(p);
        Ip_i[1] = i2_prev(p);
        xp_f[1] = i_di_to_Xi(Ip_f[1], dx2(p));
        xp_i[1] = i_di_to_Xi(Ip_i[1], dx2_prev(p));
      }
      if constexpr (D == Dim::_3D) {
        Ip_f[2] = i3(p);
        Ip_i[2] = i3_prev(p);
        xp_f[2] = i_di_to_Xi(Ip_f[2], dx3(p));
        xp_i[2] = i_di_to_Xi(Ip_i[2], dx3_prev(p));
      }
      for (auto i = 0u; i < D; ++i) {
        xp_r[i] = math::fmin(static_cast<real_t>(IMIN(Ip_i[i], Ip_f[i]) + 1),
                             math::fmax(static_cast<real_t>(IMAX(Ip_i[i], Ip_f[i])),
                                        HALF * (xp_i[i] + xp_f[i])));
      }
    }

    // Getters
    Inline void getPrtlPos(index_t& p, coord_t<M::PrtlDim>& xp) const {
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
    }

    Inline void getPrtl3Vel(index_t& p, vec_t<Dim::_3D>& vp) const {
      coord_t<M::PrtlDim> xp { ZERO };
      getPrtlPos(p, xp);
      auto inv_energy { ZERO };
      if constexpr (S == SimEngine::SRPIC) {
        metric.template transform_xyz<Idx::XYZ, Idx::U>(xp,
                                                        { ux1(p), ux2(p), ux3(p) },
                                                        vp);
        inv_energy = ONE / math::sqrt(ONE + NORM_SQR(ux1(p), ux2(p), ux3(p)));
      } else {
        metric.template transform<Idx::D, Idx::U>(xp, { ux1(p), ux2(p), ux3(p) }, vp);
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
  };

} // namespace kernel

#undef i_di_to_Xi

#endif // KERNELS_CURRENTS_DEPOSIT_HPP
