#ifndef CURRENTS_DEPOSIT_H
#define CURRENTS_DEPOSIT_H

#include "wrapper.h"

#include "particle_macros.h"

#include "io/output.h"
#include "meshblock/meshblock.h"
#include "meshblock/particles.h"
#include "utils/qmath.h"

#include <stdexcept>

namespace ntt {

  /**
   * @brief Algorithm for the current deposition.
   * @tparam D Dimension.
   */
  template <Dimension D, SimulationEngine S>
  class DepositCurrents_kernel {
    scatter_ndfield_t<D, 3> J;
    array_t<int*>           i1, i2, i3;
    array_t<int*>           i1_prev, i2_prev, i3_prev;
    array_t<prtldx_t*>      dx1, dx2, dx3;
    array_t<prtldx_t*>      dx1_prev, dx2_prev, dx3_prev;
    array_t<real_t*>        ux1, ux2, ux3;
    array_t<real_t*>        phi;
    array_t<real_t*>        weight;
    array_t<short*>         tag;
    const Metric<D>         metric;
    const real_t            charge, inv_dt;

    Inline void getPrtlPosImpl(DimensionTag<Dim1>, index_t&, coord_t<FullD>&) const;
    Inline void getPrtlPosImpl(DimensionTag<Dim2>, index_t&, coord_t<FullD>&) const;
    Inline void getPrtlPosImpl(DimensionTag<Dim3>, index_t&, coord_t<FullD>&) const;

    Inline void depositCurrentsFromParticleImpl(DimensionTag<Dim1>,
                                                const real_t&          coeff,
                                                const vec_t<Dim3>&     vp,
                                                const tuple_t<int, D>& Ip_f,
                                                const tuple_t<int, D>& Ip_i,
                                                const coord_t<D>&      xp_f,
                                                const coord_t<D>&      xp_i,
                                                const coord_t<D>& xp_r) const;
    Inline void depositCurrentsFromParticleImpl(DimensionTag<Dim2>,
                                                const real_t&          coeff,
                                                const vec_t<Dim3>&     vp,
                                                const tuple_t<int, D>& Ip_f,
                                                const tuple_t<int, D>& Ip_i,
                                                const coord_t<D>&      xp_f,
                                                const coord_t<D>&      xp_i,
                                                const coord_t<D>& xp_r) const;
    Inline void depositCurrentsFromParticleImpl(DimensionTag<Dim3>,
                                                const real_t&          coeff,
                                                const vec_t<Dim3>&     vp,
                                                const tuple_t<int, D>& Ip_f,
                                                const tuple_t<int, D>& Ip_i,
                                                const coord_t<D>&      xp_f,
                                                const coord_t<D>&      xp_i,
                                                const coord_t<D>& xp_r) const;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param particles Particles.
     * @param scatter_cur Scatter array of the currents.
     * @param charge charge of the species (code units).
     * @param dt Time step.
     */
    DepositCurrents_kernel(const Meshblock<D, S>&         mblock,
                           const Particles<D, S>&         particles,
                           const scatter_ndfield_t<D, 3>& scatter_cur,
                           const real_t&                  charge,
                           const real_t&                  dt) :
      J { scatter_cur },
      i1 { particles.i1 },
      i2 { particles.i2 },
      i3 { particles.i3 },
      i1_prev { particles.i1_prev },
      i2_prev { particles.i2_prev },
      i3_prev { particles.i3_prev },
      dx1 { particles.dx1 },
      dx2 { particles.dx2 },
      dx3 { particles.dx3 },
      dx1_prev { particles.dx1_prev },
      dx2_prev { particles.dx2_prev },
      dx3_prev { particles.dx3_prev },
      ux1 { particles.ux1 },
      ux2 { particles.ux2 },
      ux3 { particles.ux3 },
      phi { particles.phi },
      weight { particles.weight },
      tag { particles.tag },
      metric { mblock.metric },
      charge { charge },
      inv_dt { ONE / dt } {}

    /**
     * @brief Iteration of the loop over particles.
     * @param p index.
     */
    Inline auto operator()(index_t p) const -> void {
      if (tag(p) == ParticleTag::alive) {
        // _f = final, _i = initial
        tuple_t<int, D> Ip_f, Ip_i;
        coord_t<D>      xp_f, xp_i, xp_r;
        vec_t<Dim3>     vp { ZERO };

        // get [i, di]_init and [i, di]_final (per dimension)
        getDepositInterval(p, Ip_f, Ip_i, xp_f, xp_i, xp_r);
        // recover particle velocity to deposit in unsimulated direction
        getPrtl3Vel(p, vp);
        const auto coeff = weight(p) * charge;
        depositCurrentsFromParticle(coeff, vp, Ip_f, Ip_i, xp_f, xp_i, xp_r);
      }
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
                                            const vec_t<Dim3>&     vp,
                                            const tuple_t<int, D>& Ip_f,
                                            const tuple_t<int, D>& Ip_i,
                                            const coord_t<D>&      xp_f,
                                            const coord_t<D>&      xp_i,
                                            const coord_t<D>& xp_r) const -> void {
      depositCurrentsFromParticleImpl(DimensionTag<D> {},
                                      coeff,
                                      vp,
                                      Ip_f,
                                      Ip_i,
                                      xp_f,
                                      xp_i,
                                      xp_r);
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
      if constexpr (D == Dim2 || D == Dim3) {
        Ip_f[1] = i2(p);
        Ip_i[1] = i2_prev(p);
        xp_f[1] = i_di_to_Xi(Ip_f[1], dx2(p));
        xp_i[1] = i_di_to_Xi(Ip_i[1], dx2_prev(p));
      }
      if constexpr (D == Dim3) {
        Ip_f[2] = i3(p);
        Ip_i[2] = i3_prev(p);
        xp_f[2] = i_di_to_Xi(Ip_f[2], dx3(p));
        xp_i[2] = i_di_to_Xi(Ip_i[2], dx3_prev(p));
      }
      for (short i { 0 }; i < static_cast<short>(D); ++i) {
        xp_r[i] = math::fmin(static_cast<real_t>(IMIN(Ip_i[i], Ip_f[i]) + 1),
                             math::fmax(static_cast<real_t>(IMAX(Ip_i[i], Ip_f[i])),
                                        HALF * (xp_i[i] + xp_f[i])));
      }
    }

    // Getters
    Inline void getPrtlPos(index_t& p, coord_t<FullD>& xp) const {
      getPrtlPosImpl(DimensionTag<D> {}, p, xp);
    }

    Inline void getPrtl3Vel(index_t& p, vec_t<Dim3>& vp) const {
      coord_t<FullD> xp { ZERO };
      getPrtlPos(p, xp);
      auto inv_energy { ZERO };
      if constexpr (S != GRPICEngine) {
        metric.v3_Cart2Cntrv(xp, { ux1(p), ux2(p), ux3(p) }, vp);
        inv_energy = ONE / math::sqrt(ONE + NORM_SQR(ux1(p), ux2(p), ux3(p)));
      } else {
        metric.v3_Cov2Cntrv(xp, { ux1(p), ux2(p), ux3(p) }, vp);
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

  /**
   * !PERFORM: fix the conversion to I+di
   */
  template <Dimension D, SimulationEngine S>
  Inline void DepositCurrents_kernel<D, S>::depositCurrentsFromParticleImpl(
    DimensionTag<Dim1>,
    const real_t&          coeff,
    const vec_t<Dim3>&     vp,
    const tuple_t<int, D>& Ip_f,
    const tuple_t<int, D>& Ip_i,
    const coord_t<D>&      xp_f,
    const coord_t<D>&      xp_i,
    const coord_t<D>&      xp_r) const {
    const real_t Wx1_1 { HALF * (xp_i[0] + xp_r[0]) - static_cast<real_t>(Ip_i[0]) };
    const real_t Wx1_2 { HALF * (xp_f[0] + xp_r[0]) - static_cast<real_t>(Ip_f[0]) };
    const real_t Fx1_1 { (xp_r[0] - xp_i[0]) * coeff * inv_dt };
    const real_t Fx1_2 { (xp_f[0] - xp_r[0]) * coeff * inv_dt };

    const real_t Fx2_1 { HALF * vp[1] * coeff };
    const real_t Fx2_2 { HALF * vp[1] * coeff };

    const real_t Fx3_1 { HALF * vp[2] * coeff };
    const real_t Fx3_2 { HALF * vp[2] * coeff };

    auto J_acc                           = J.access();
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
  }

  template <Dimension D, SimulationEngine S>
  Inline void DepositCurrents_kernel<D, S>::depositCurrentsFromParticleImpl(
    DimensionTag<Dim2>,
    const real_t&          coeff,
    const vec_t<Dim3>&     vp,
    const tuple_t<int, D>& Ip_f,
    const tuple_t<int, D>& Ip_i,
    const coord_t<D>&      xp_f,
    const coord_t<D>&      xp_i,
    const coord_t<D>&      xp_r) const {
    const real_t Wx1_1 { HALF * (xp_i[0] + xp_r[0]) - static_cast<real_t>(Ip_i[0]) };
    const real_t Wx1_2 { HALF * (xp_f[0] + xp_r[0]) - static_cast<real_t>(Ip_f[0]) };
    const real_t Fx1_1 { (xp_r[0] - xp_i[0]) * coeff * inv_dt };
    const real_t Fx1_2 { (xp_f[0] - xp_r[0]) * coeff * inv_dt };

    const real_t Wx2_1 { HALF * (xp_i[1] + xp_r[1]) - static_cast<real_t>(Ip_i[1]) };
    const real_t Wx2_2 { HALF * (xp_f[1] + xp_r[1]) - static_cast<real_t>(Ip_f[1]) };
    const real_t Fx2_1 { (xp_r[1] - xp_i[1]) * coeff * inv_dt };
    const real_t Fx2_2 { (xp_f[1] - xp_r[1]) * coeff * inv_dt };

    const real_t Fx3_1 { HALF * vp[2] * coeff };
    const real_t Fx3_2 { HALF * vp[2] * coeff };

#ifdef DEBUG
    auto nan_found = false;
    if (Kokkos::isnan(Wx1_1) || Kokkos::isinf(Wx1_1)) {
      nan_found = true;
      printf("NAN found in Wx1_1 ");
    }
    if (Kokkos::isnan(Wx1_2) || Kokkos::isinf(Wx1_2)) {
      nan_found = true;
      printf("NAN found in Wx1_2 ");
    }
    if (Kokkos::isnan(Fx1_1) || Kokkos::isinf(Fx1_1)) {
      nan_found = true;
      printf("NAN found in Fx1_1 ");
    }
    if (Kokkos::isnan(Fx1_2) || Kokkos::isinf(Fx1_2)) {
      nan_found = true;
      printf("NAN found in Fx1_2 ");
    }
    if (Kokkos::isnan(Wx2_1) || Kokkos::isinf(Wx2_1)) {
      nan_found = true;
      printf("NAN found in Wx2_1 ");
    }
    if (Kokkos::isnan(Wx2_2) || Kokkos::isinf(Wx2_2)) {
      nan_found = true;
      printf("NAN found in Wx2_2 ");
    }
    if (Kokkos::isnan(Fx2_1) || Kokkos::isinf(Fx2_1)) {
      nan_found = true;
      printf("NAN found in Fx2_1 ");
    }
    if (Kokkos::isnan(Fx2_2) || Kokkos::isinf(Fx2_2)) {
      nan_found = true;
      printf("NAN found in Fx2_2 ");
    }
    if (Kokkos::isnan(Fx3_1) || Kokkos::isinf(Fx3_1)) {
      nan_found = true;
      printf("NAN found in Fx3_1 ");
    }
    if (Kokkos::isnan(Fx3_2) || Kokkos::isinf(Fx3_2)) {
      nan_found = true;
      printf("NAN found in Fx3_2 ");
    }
    if (nan_found) {
      printf("prtl: xp_f=%f %f xp_i=%f %f xp_r=%f %f vp=%f %f %f\n",
             xp_f[0],
             xp_f[1],
             xp_i[0],
             xp_i[1],
             xp_r[0],
             xp_r[1],
             vp[0],
             vp[1],
             vp[2]);
    }
#endif

    auto J_acc                                               = J.access();
    J_acc(Ip_i[0] + N_GHOSTS, Ip_i[1] + N_GHOSTS, cur::jx1) += Fx1_1 *
                                                               (ONE - Wx2_1);
    J_acc(Ip_i[0] + N_GHOSTS, Ip_i[1] + N_GHOSTS + 1, cur::jx1) += Fx1_1 * Wx2_1;
    J_acc(Ip_f[0] + N_GHOSTS, Ip_f[1] + N_GHOSTS, cur::jx1) += Fx1_2 *
                                                               (ONE - Wx2_2);
    J_acc(Ip_f[0] + N_GHOSTS, Ip_f[1] + N_GHOSTS + 1, cur::jx1) += Fx1_2 * Wx2_2;

    J_acc(Ip_i[0] + N_GHOSTS, Ip_i[1] + N_GHOSTS, cur::jx2) += Fx2_1 *
                                                               (ONE - Wx1_1);
    J_acc(Ip_i[0] + N_GHOSTS + 1, Ip_i[1] + N_GHOSTS, cur::jx2) += Fx2_1 * Wx1_1;
    J_acc(Ip_f[0] + N_GHOSTS, Ip_f[1] + N_GHOSTS, cur::jx2) += Fx2_2 *
                                                               (ONE - Wx1_2);
    J_acc(Ip_f[0] + N_GHOSTS + 1, Ip_f[1] + N_GHOSTS, cur::jx2) += Fx2_2 * Wx1_2;

    J_acc(Ip_i[0] + N_GHOSTS, Ip_i[1] + N_GHOSTS, cur::jx3) += Fx3_1 *
                                                               (ONE - Wx1_1) *
                                                               (ONE - Wx2_1);
    J_acc(Ip_i[0] + N_GHOSTS + 1, Ip_i[1] + N_GHOSTS, cur::jx3) += Fx3_1 * Wx1_2 *
                                                                   (ONE - Wx2_1);
    J_acc(Ip_i[0] + N_GHOSTS, Ip_i[1] + N_GHOSTS + 1, cur::jx3) += Fx3_1 *
                                                                   (ONE - Wx1_1) *
                                                                   Wx2_1;
    J_acc(Ip_i[0] + N_GHOSTS + 1, Ip_i[1] + N_GHOSTS + 1, cur::jx3) += Fx3_1 *
                                                                       Wx1_1 *
                                                                       Wx2_1;

    J_acc(Ip_f[0] + N_GHOSTS, Ip_f[1] + N_GHOSTS, cur::jx3) += Fx3_2 *
                                                               (ONE - Wx1_2) *
                                                               (ONE - Wx2_2);
    J_acc(Ip_f[0] + N_GHOSTS + 1, Ip_f[1] + N_GHOSTS, cur::jx3) += Fx3_2 * Wx1_2 *
                                                                   (ONE - Wx2_2);
    J_acc(Ip_f[0] + N_GHOSTS, Ip_f[1] + N_GHOSTS + 1, cur::jx3) += Fx3_2 *
                                                                   (ONE - Wx1_2) *
                                                                   Wx2_2;
    J_acc(Ip_f[0] + N_GHOSTS + 1, Ip_f[1] + N_GHOSTS + 1, cur::jx3) += Fx3_2 *
                                                                       Wx1_2 *
                                                                       Wx2_2;
  }

  template <Dimension D, SimulationEngine S>
  Inline void DepositCurrents_kernel<D, S>::depositCurrentsFromParticleImpl(
    DimensionTag<Dim3>,
    const real_t&          coeff,
    const vec_t<Dim3>&     vp,
    const tuple_t<int, D>& Ip_f,
    const tuple_t<int, D>& Ip_i,
    const coord_t<D>&      xp_f,
    const coord_t<D>&      xp_i,
    const coord_t<D>&      xp_r) const {
    const real_t Wx1_1 { HALF * (xp_i[0] + xp_r[0]) - static_cast<real_t>(Ip_i[0]) };
    const real_t Wx1_2 { HALF * (xp_f[0] + xp_r[0]) - static_cast<real_t>(Ip_f[0]) };
    const real_t Fx1_1 { (xp_r[0] - xp_i[0]) * coeff * inv_dt };
    const real_t Fx1_2 { (xp_f[0] - xp_r[0]) * coeff * inv_dt };

    const real_t Wx2_1 { HALF * (xp_i[1] + xp_r[1]) - static_cast<real_t>(Ip_i[1]) };
    const real_t Wx2_2 { HALF * (xp_f[1] + xp_r[1]) - static_cast<real_t>(Ip_f[1]) };
    const real_t Fx2_1 { (xp_r[1] - xp_i[1]) * coeff * inv_dt };
    const real_t Fx2_2 { (xp_f[1] - xp_r[1]) * coeff * inv_dt };

    const real_t Wx3_1 { HALF * (xp_i[2] + xp_r[2]) - static_cast<real_t>(Ip_i[2]) };
    const real_t Wx3_2 { HALF * (xp_f[2] + xp_r[2]) - static_cast<real_t>(Ip_f[2]) };
    const real_t Fx3_1 { (xp_r[2] - xp_i[2]) * coeff * inv_dt };
    const real_t Fx3_2 { (xp_f[2] - xp_r[2]) * coeff * inv_dt };

    auto J_acc       = J.access();
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

#ifdef MINKOWSKI_METRIC
  template <Dimension D, SimulationEngine S>
  Inline void DepositCurrents_kernel<D, S>::getPrtlPosImpl(DimensionTag<Dim1>,
                                                           index_t& p,
                                                           coord_t<D>& xp) const {
    xp[0] = i_di_to_Xi(i1(p), dx1(p));
  }

  template <Dimension D, SimulationEngine S>
  Inline void DepositCurrents_kernel<D, S>::getPrtlPosImpl(DimensionTag<Dim2>,
                                                           index_t& p,
                                                           coord_t<D>& xp) const {
    xp[0] = i_di_to_Xi(i1(p), dx1(p));
    xp[1] = i_di_to_Xi(i2(p), dx2(p));
  }
#else
  template <Dimension D, SimulationEngine S>
  Inline void DepositCurrents_kernel<D, S>::getPrtlPosImpl(DimensionTag<Dim1>,
                                                           index_t&,
                                                           coord_t<FullD>&) const {
    NTTError("not applicable");
  }

  template <Dimension D, SimulationEngine S>
  Inline void DepositCurrents_kernel<D, S>::getPrtlPosImpl(DimensionTag<Dim2>,
                                                           index_t& p,
                                                           coord_t<FullD>& xp) const {
    xp[0] = i_di_to_Xi(i1(p), dx1(p));
    xp[1] = i_di_to_Xi(i2(p), dx2(p));
    xp[2] = phi(p);
  }
#endif

  template <Dimension D, SimulationEngine S>
  Inline void DepositCurrents_kernel<D, S>::getPrtlPosImpl(DimensionTag<Dim3>,
                                                           index_t& p,
                                                           coord_t<FullD>& xp) const {
    xp[0] = i_di_to_Xi(i1(p), dx1(p));
    xp[1] = i_di_to_Xi(i2(p), dx2(p));
    xp[2] = i_di_to_Xi(i3(p), dx3(p));
  }

} // namespace ntt

#endif // CURRENTS_DEPOSIT_H