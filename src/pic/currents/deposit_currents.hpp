#ifndef PIC_DEPOSIT_CURRENTS_H
#define PIC_DEPOSIT_CURRENTS_H

#include "wrapper.h"

#include "field_macros.h"
#include "fields.h"
#include "meshblock.h"
#include "particle_macros.h"
#include "particles.h"
#include "pic.h"
#include "qmath.h"

#include <stdexcept>

namespace ntt {

  /**
   * @brief Algorithm for the current deposition.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class DepositCurrents_kernel {
    Meshblock<D, PICEngine> m_mblock;
    Particles<D, PICEngine> m_particles;
    scatter_ndfield_t<D, 3> m_scatter_cur;
    const real_t            m_charge, m_dt;
    const real_t            m_xi2max;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param particles Particles.
     * @param scatter_cur Scatter array of the currents.
     * @param charge charge of the species (code units).
     * @param dt Time step.
     */
    DepositCurrents_kernel(const Meshblock<D, PICEngine>& mblock,
                           const Particles<D, PICEngine>& particles,
                           const scatter_ndfield_t<D, 3>& scatter_cur,
                           const real_t&                  charge,
                           const real_t&                  dt)
      : m_mblock(mblock),
        m_particles(particles),
        m_scatter_cur(scatter_cur),
        m_charge(charge),
        m_dt(dt),
        m_xi2max((real_t)(m_mblock.i2_max()) - (real_t)(N_GHOSTS)) {}

    /**
     * @brief Iteration of the loop over particles.
     * @param p index.
     */
    Inline void operator()(index_t p) const {
      if (!m_particles.is_dead(p)) {
        // _f = final, _i = initial
        tuple_t<int, D> Ip_f, Ip_i;
        coord_t<D>      xp_f, xp_i, xp_r;
        vec_t<Dim3>     vp { ZERO, ZERO, ZERO };

        // get [i, di]_init and [i, di]_final (per dimension)
        getDepositInterval(p, vp, Ip_f, Ip_i, xp_f, xp_i, xp_r);
        depositCurrentsFromParticle(vp, Ip_f, Ip_i, xp_f, xp_i, xp_r);
      }
    }

    /**
     * @brief Deposit currents from a single particle.
     * @param[in] vp Particle 3-velocity.
     * @param[in] Ip_f Final position of the particle (cell index).
     * @param[in] Ip_i Initial position of the particle (cell index).
     * @param[in] xp_f Final position.
     * @param[in] xp_i Previous step position.
     * @param[in] xp_r Intermediate point used in zig-zag deposit.
     */
    Inline void depositCurrentsFromParticle(const vec_t<Dim3>&     vp,
                                            const tuple_t<int, D>& Ip_f,
                                            const tuple_t<int, D>& Ip_i,
                                            const coord_t<D>&      xp_f,
                                            const coord_t<D>&      xp_i,
                                            const coord_t<D>&      xp_r) const;

    /**
     * @brief Get particle position in `coord_t` form.
     * @param[in] p Index of particle.
     * @param[out] vp Particle 3-velocity.
     * @param[out] Ip_f Final position of the particle (cell index).
     * @param[out] Ip_i Initial position of the particle (cell index).
     * @param[out] xp_f Final position.
     * @param[out] xp_i Previous step position.
     * @param[out] xp_r Intermediate point used in zig-zag deposit.
     */
    Inline void getDepositInterval(index_t&         p,
                                   vec_t<Dim3>&     vp,
                                   tuple_t<int, D>& Ip_f,
                                   tuple_t<int, D>& Ip_i,
                                   coord_t<D>&      xp_f,
                                   coord_t<D>&      xp_i,
                                   coord_t<D>&      xp_r) const {
      real_t            inv_energy;
      tuple_t<float, D> dIp_f;

      if constexpr ((D == Dim1) || (D == Dim2) || (D == Dim3)) {
        Ip_f[0]  = m_particles.i1(p);
        dIp_f[0] = m_particles.dx1(p);
      }

      if constexpr ((D == Dim2) || (D == Dim3)) {
        Ip_f[1]  = m_particles.i2(p);
        dIp_f[1] = m_particles.dx2(p);
      }

      if constexpr (D == Dim3) {
        Ip_f[2]  = m_particles.i3(p);
        dIp_f[2] = m_particles.dx3(p);
      }

      for (short i { 0 }; i < static_cast<short>(D); ++i) {
        xp_f[i] = static_cast<real_t>(Ip_f[i]) + static_cast<real_t>(dIp_f[i]);
      }

#ifdef MINKOWSKI_METRIC
      m_mblock.metric.v_Cart2Cntrv(
        xp_f, { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) }, vp);
#else
      coord_t<Dim3> xp;
      xp[0] = xp_f[0];
      xp[1] = xp_f[1];
      xp[2] = m_particles.phi(p);
      m_mblock.metric.v_Cart2Cntrv(
        xp, { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) }, vp);
      if constexpr (D == Dim2) {
        if (Ip_f[1] == 0 && AlmostEqual(dIp_f[1], 0.0f)) {
          vp[2] = ZERO;
        } else if (Ip_f[1] == static_cast<int>(m_xi2max) - 1 && AlmostEqual(dIp_f[1], 1.0f)) {
          vp[2] = ZERO;
        }
      }
#endif
      inv_energy = ONE / get_prtl_Gamma_SR(m_particles, p);

      // get particle 3-velocity in coordinate basis
      for (short i { 0 }; i < 3; ++i) {
        vp[i] *= inv_energy;
      }

      for (short i { 0 }; i < static_cast<short>(D); ++i) {
        int I_i;
        xp_i[i] = xp_f[i] - m_dt * vp[i];
        from_Xi_to_i(xp_i[i], I_i);

#ifndef MINKOWSKI_METRIC
        if constexpr (D == Dim2) {
          if (i == 1) {
            const bool northern_pole  = (I_i < 0);
            const bool sourthern_pole = (I_i >= static_cast<int>(m_xi2max));
            if (northern_pole || sourthern_pole) {
              I_i     = northern_pole ? 0 : static_cast<int>(m_xi2max) - 1;
              xp_i[i] = northern_pole ? -xp_i[i] : (TWO * m_xi2max - xp_i[i]);
            }
          }
        }
#endif
        Ip_i[i]             = I_i;
        const real_t xi_mid = HALF * (xp_i[i] + xp_f[i]);
        xp_r[i]
          = math::fmin(static_cast<real_t>(math::fmin(Ip_i[i], Ip_f[i]) + 1),
                       math::fmax(static_cast<real_t>(math::fmax(Ip_i[i], Ip_f[i])), xi_mid));
      }
    }
  };

  template <>
  Inline void DepositCurrents_kernel<Dim1>::depositCurrentsFromParticle(
    const vec_t<Dim3>&        vp,
    const tuple_t<int, Dim1>& Ip_f,
    const tuple_t<int, Dim1>& Ip_i,
    const coord_t<Dim1>&      xp_f,
    const coord_t<Dim1>&      xp_i,
    const coord_t<Dim1>&      xp_r) const {
    real_t Wx1_1 { HALF * (xp_i[0] + xp_r[0]) - static_cast<real_t>(Ip_i[0]) };
    real_t Wx1_2 { HALF * (xp_f[0] + xp_r[0]) - static_cast<real_t>(Ip_f[0]) };
    real_t Fx1_1 { (xp_r[0] - xp_i[0]) * m_charge / m_dt };
    real_t Fx1_2 { (xp_f[0] - xp_r[0]) * m_charge / m_dt };

    real_t Fx2_1 { HALF * vp[1] * m_charge };
    real_t Fx2_2 { HALF * vp[1] * m_charge };

    real_t Fx3_1 { HALF * vp[2] * m_charge };
    real_t Fx3_2 { HALF * vp[2] * m_charge };

    auto   cur_access = m_scatter_cur.access();
    ATOMIC_JX1(Ip_i[0]) += Fx1_1;
    ATOMIC_JX1(Ip_f[0]) += Fx1_2;

    ATOMIC_JX2(Ip_i[0]) += Fx2_1 * (ONE - Wx1_1);
    ATOMIC_JX2(Ip_i[0] + 1) += Fx2_1 * Wx1_1;
    ATOMIC_JX2(Ip_f[0]) += Fx2_2 * (ONE - Wx1_2);
    ATOMIC_JX2(Ip_f[0] + 1) += Fx2_2 * Wx1_2;

    ATOMIC_JX3(Ip_i[0]) += Fx3_1 * (ONE - Wx1_1);
    ATOMIC_JX3(Ip_i[0] + 1) += Fx3_1 * Wx1_1;
    ATOMIC_JX3(Ip_f[0]) += Fx3_2 * (ONE - Wx1_2);
    ATOMIC_JX3(Ip_f[0] + 1) += Fx3_2 * Wx1_2;
  }

  /**
   * !TODO: fix the conversion to I+di
   */
  template <>
  Inline void DepositCurrents_kernel<Dim2>::depositCurrentsFromParticle(
    const vec_t<Dim3>&        vp,
    const tuple_t<int, Dim2>& Ip_f,
    const tuple_t<int, Dim2>& Ip_i,
    const coord_t<Dim2>&      xp_f,
    const coord_t<Dim2>&      xp_i,
    const coord_t<Dim2>&      xp_r) const {
    real_t Wx1_1 { HALF * (xp_i[0] + xp_r[0]) - static_cast<real_t>(Ip_i[0]) };
    real_t Wx1_2 { HALF * (xp_f[0] + xp_r[0]) - static_cast<real_t>(Ip_f[0]) };
    real_t Fx1_1 { (xp_r[0] - xp_i[0]) * m_charge / m_dt };
    real_t Fx1_2 { (xp_f[0] - xp_r[0]) * m_charge / m_dt };

    real_t Wx2_1 { HALF * (xp_i[1] + xp_r[1]) - static_cast<real_t>(Ip_i[1]) };
    real_t Wx2_2 { HALF * (xp_f[1] + xp_r[1]) - static_cast<real_t>(Ip_f[1]) };
    real_t Fx2_1 { (xp_r[1] - xp_i[1]) * m_charge / m_dt };
    real_t Fx2_2 { (xp_f[1] - xp_r[1]) * m_charge / m_dt };

    real_t Fx3_1 { HALF * vp[2] * m_charge };
    real_t Fx3_2 { HALF * vp[2] * m_charge };

    auto   cur_access = m_scatter_cur.access();
    ATOMIC_JX1(Ip_i[0], Ip_i[1]) += Fx1_1 * (ONE - Wx2_1);
    ATOMIC_JX1(Ip_i[0], Ip_i[1] + 1) += Fx1_1 * Wx2_1;
    ATOMIC_JX1(Ip_f[0], Ip_f[1]) += Fx1_2 * (ONE - Wx2_2);
    ATOMIC_JX1(Ip_f[0], Ip_f[1] + 1) += Fx1_2 * Wx2_2;

    ATOMIC_JX2(Ip_i[0], Ip_i[1]) += Fx2_1 * (ONE - Wx1_1);
    ATOMIC_JX2(Ip_i[0] + 1, Ip_i[1]) += Fx2_1 * Wx1_1;
    ATOMIC_JX2(Ip_f[0], Ip_f[1]) += Fx2_2 * (ONE - Wx1_2);
    ATOMIC_JX2(Ip_f[0] + 1, Ip_f[1]) += Fx2_2 * Wx1_2;

    ATOMIC_JX3(Ip_i[0], Ip_i[1]) += Fx3_1 * (ONE - Wx1_1) * (ONE - Wx2_1);
    ATOMIC_JX3(Ip_i[0] + 1, Ip_i[1]) += Fx3_1 * Wx1_2 * (ONE - Wx2_1);
    ATOMIC_JX3(Ip_i[0], Ip_i[1] + 1) += Fx3_1 * (ONE - Wx1_1) * Wx2_1;
    ATOMIC_JX3(Ip_i[0] + 1, Ip_i[1] + 1) += Fx3_1 * Wx1_1 * Wx2_1;

    ATOMIC_JX3(Ip_f[0], Ip_f[1]) += Fx3_2 * (ONE - Wx1_2) * (ONE - Wx2_2);
    ATOMIC_JX3(Ip_f[0] + 1, Ip_f[1]) += Fx3_2 * Wx1_2 * (ONE - Wx2_2);
    ATOMIC_JX3(Ip_f[0], Ip_f[1] + 1) += Fx3_2 * (ONE - Wx1_2) * Wx2_2;
    ATOMIC_JX3(Ip_f[0] + 1, Ip_f[1] + 1) += Fx3_2 * Wx1_2 * Wx2_2;
  }

  template <>
  Inline void DepositCurrents_kernel<Dim3>::depositCurrentsFromParticle(
    const vec_t<Dim3>&,
    const tuple_t<int, Dim3>& Ip_f,
    const tuple_t<int, Dim3>& Ip_i,
    const coord_t<Dim3>&      xp_f,
    const coord_t<Dim3>&      xp_i,
    const coord_t<Dim3>&      xp_r) const {
    real_t Wx1_1 { HALF * (xp_i[0] + xp_r[0]) - static_cast<real_t>(Ip_i[0]) };
    real_t Wx1_2 { HALF * (xp_f[0] + xp_r[0]) - static_cast<real_t>(Ip_f[0]) };
    real_t Fx1_1 { (xp_r[0] - xp_i[0]) * m_charge / m_dt };
    real_t Fx1_2 { (xp_f[0] - xp_r[0]) * m_charge / m_dt };

    real_t Wx2_1 { HALF * (xp_i[1] + xp_r[1]) - static_cast<real_t>(Ip_i[1]) };
    real_t Wx2_2 { HALF * (xp_f[1] + xp_r[1]) - static_cast<real_t>(Ip_f[1]) };
    real_t Fx2_1 { (xp_r[1] - xp_i[1]) * m_charge / m_dt };
    real_t Fx2_2 { (xp_f[1] - xp_r[1]) * m_charge / m_dt };

    real_t Wx3_1 { HALF * (xp_i[2] + xp_r[2]) - static_cast<real_t>(Ip_i[2]) };
    real_t Wx3_2 { HALF * (xp_f[2] + xp_r[2]) - static_cast<real_t>(Ip_f[2]) };
    real_t Fx3_1 { (xp_r[2] - xp_i[2]) * m_charge / m_dt };
    real_t Fx3_2 { (xp_f[2] - xp_r[2]) * m_charge / m_dt };

    auto   cur_access = m_scatter_cur.access();
    ATOMIC_JX1(Ip_i[0], Ip_i[1], Ip_i[2]) += Fx1_1 * (ONE - Wx2_1) * (ONE - Wx3_1);
    ATOMIC_JX1(Ip_i[0], Ip_i[1] + 1, Ip_i[2]) += Fx1_1 * Wx2_1 * (ONE - Wx3_1);
    ATOMIC_JX1(Ip_i[0], Ip_i[1], Ip_i[2] + 1) += Fx1_1 * (ONE - Wx2_1) * Wx3_1;
    ATOMIC_JX1(Ip_i[0], Ip_i[1] + 1, Ip_i[2] + 1) += Fx1_1 * Wx2_1 * Wx3_1;

    ATOMIC_JX1(Ip_f[0], Ip_f[1], Ip_f[2]) += Fx1_2 * (ONE - Wx2_2) * (ONE - Wx3_2);
    ATOMIC_JX1(Ip_f[0], Ip_f[1] + 1, Ip_f[2]) += Fx1_2 * Wx2_2 * (ONE - Wx3_2);
    ATOMIC_JX1(Ip_f[0], Ip_f[1], Ip_f[2] + 1) += Fx1_2 * (ONE - Wx2_2) * Wx3_2;
    ATOMIC_JX1(Ip_f[0], Ip_f[1] + 1, Ip_f[2] + 1) += Fx1_2 * Wx2_2 * Wx3_2;

    ATOMIC_JX2(Ip_i[0], Ip_i[1], Ip_i[2]) += Fx2_1 * (ONE - Wx1_1) * (ONE - Wx3_1);
    ATOMIC_JX2(Ip_i[0] + 1, Ip_i[1], Ip_i[2]) += Fx2_1 * Wx1_1 * (ONE - Wx3_1);
    ATOMIC_JX2(Ip_i[0], Ip_i[1], Ip_i[2] + 1) += Fx2_1 * (ONE - Wx1_1) * Wx3_1;
    ATOMIC_JX2(Ip_i[0] + 1, Ip_i[1], Ip_i[2] + 1) += Fx2_1 * Wx1_1 * Wx3_1;

    ATOMIC_JX2(Ip_f[0], Ip_f[1], Ip_f[2]) += Fx2_2 * (ONE - Wx1_2) * (ONE - Wx3_2);
    ATOMIC_JX2(Ip_f[0] + 1, Ip_f[1], Ip_f[2]) += Fx2_2 * Wx1_2 * (ONE - Wx3_2);
    ATOMIC_JX2(Ip_f[0], Ip_f[1], Ip_f[2] + 1) += Fx2_2 * (ONE - Wx1_2) * Wx3_2;
    ATOMIC_JX2(Ip_f[0] + 1, Ip_f[1], Ip_f[2] + 1) += Fx2_2 * Wx1_2 * Wx3_2;

    ATOMIC_JX3(Ip_i[0], Ip_i[1], Ip_i[2]) += Fx3_1 * (ONE - Wx1_1) * (ONE - Wx2_1);
    ATOMIC_JX3(Ip_i[0] + 1, Ip_i[1], Ip_i[2]) += Fx3_1 * Wx1_1 * (ONE - Wx2_1);
    ATOMIC_JX3(Ip_i[0], Ip_i[1] + 1, Ip_i[2]) += Fx3_1 * (ONE - Wx1_1) * Wx2_1;
    ATOMIC_JX3(Ip_i[0] + 1, Ip_i[1] + 1, Ip_i[2]) += Fx3_1 * Wx1_1 * Wx2_1;

    ATOMIC_JX3(Ip_f[0], Ip_f[1], Ip_f[2]) += Fx3_2 * (ONE - Wx1_2) * (ONE - Wx2_2);
    ATOMIC_JX3(Ip_f[0] + 1, Ip_f[1], Ip_f[2]) += Fx3_2 * Wx1_2 * (ONE - Wx2_2);
    ATOMIC_JX3(Ip_f[0], Ip_f[1] + 1, Ip_f[2]) += Fx3_2 * (ONE - Wx1_2) * Wx2_2;
    ATOMIC_JX3(Ip_f[0] + 1, Ip_f[1] + 1, Ip_f[2]) += Fx3_2 * Wx1_2 * Wx2_2;
  }

}    // namespace ntt

#endif
