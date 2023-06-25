#ifndef GRPIC_CURRENTS_DEPOSIT_H
#define GRPIC_CURRENTS_DEPOSIT_H

#include "wrapper.h"

#include "field_macros.h"
#include "grpic.h"
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
  template <Dimension D>
  class DepositCurrents_kernel {
    Meshblock<D, GRPICEngine> m_mblock;
    Particles<D, GRPICEngine> m_particles;
    scatter_ndfield_t<D, 3>   m_scatter_cur0;
    const real_t              m_charge, m_dt;
    const real_t              m_xi2max;
    const bool                m_use_weights;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param particles Particles.
     * @param scatter_cur Scatter array of the currents.
     * @param charge charge of the species (code units).
     * @param dt Time step.
     */
    DepositCurrents_kernel(const Meshblock<D, GRPICEngine>& mblock,
                           const Particles<D, GRPICEngine>& particles,
                           const scatter_ndfield_t<D, 3>&   scatter_cur0,
                           const real_t&                    charge,
                           const bool&                      use_weights,
                           const real_t&                    dt)
      : m_mblock { mblock },
        m_particles { particles },
        m_scatter_cur0 { scatter_cur0 },
        m_charge { charge },
        m_use_weights { use_weights },
        m_dt { dt },
        m_xi2max { (real_t)(m_mblock.i2_max()) - (real_t)(N_GHOSTS) } {}

    /**
     * @brief Iteration of the loop over particles.
     * @param p index.
     */
    Inline void operator()(index_t p) const {
      if (m_particles.tag(p) == static_cast<short>(ParticleTag::alive)) {
        // _f = final, _i = initial
        tuple_t<int, D> Ip_f, Ip_i;
        coord_t<D>      xp_f, xp_i, xp_r;
        vec_t<Dim3>     vp { ZERO, ZERO, ZERO };

        // get [i, di]_init and [i, di]_final (per dimension)
        getDepositInterval(p, vp, Ip_f, Ip_i, xp_f, xp_i, xp_r);
        depositCurrentsFromParticle(m_use_weights ? static_cast<real_t>(m_particles.weight(p))
                                                  : ONE,
                                    vp,
                                    Ip_f,
                                    Ip_i,
                                    xp_f,
                                    xp_i,
                                    xp_r);
      }
    }

    /**
     * @brief Deposit currents from a single particle.
     * @param[in] weight Particle weight or 1 if no weights used.
     * @param[in] vp Particle 3-velocity.
     * @param[in] Ip_f Final position of the particle (cell index).
     * @param[in] Ip_i Initial position of the particle (cell index).
     * @param[in] xp_f Final position.
     * @param[in] xp_i Previous step position.
     * @param[in] xp_r Intermediate point used in zig-zag deposit.
     */
    Inline void depositCurrentsFromParticle(const real_t&          weight,
                                            const vec_t<Dim3>&     vp,
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
      tuple_t<prtldx_t, D> dIp_f;

      Ip_f[0]  = m_particles.i1(p);
      dIp_f[0] = m_particles.dx1(p);

      Ip_f[1]  = m_particles.i2(p);
      dIp_f[1] = m_particles.dx2(p);

      if constexpr (D == Dim3) {
        Ip_f[2]  = m_particles.i3(p);
        dIp_f[2] = m_particles.dx3(p);
      }

      for (short i { 0 }; i < static_cast<short>(D); ++i) {
        xp_f[i] = static_cast<real_t>(Ip_f[i]) + static_cast<real_t>(dIp_f[i]);
      }

      // find 3-velocity
      vec_t<Dim3> u_cntrv { ZERO };
      m_mblock.metric.v3_Cov2Cntrv(
        xp_f, { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) }, u_cntrv);
      const auto gamma
        = math::sqrt(ONE + m_particles.ux1(p) * u_cntrv[0] + m_particles.ux2(p) * u_cntrv[1]
                     + m_particles.ux3(p) * u_cntrv[2]);

      vp[0]   = m_particles.ux1(p) / gamma;
      vp[1]   = m_particles.ux2(p) / gamma;
      vp[2]   = m_particles.ux3(p) / gamma;

      // !Q: no alpha here, right?
      Ip_i[0] = m_particles.i1_prev(p);
      xp_i[0] = static_cast<real_t>(m_particles.i1_prev(p))
                + static_cast<real_t>(m_particles.dx1_prev(p));
      Ip_i[1] = m_particles.i2_prev(p);
      xp_i[1] = static_cast<real_t>(m_particles.i2_prev(p))
                + static_cast<real_t>(m_particles.dx2_prev(p));

      if constexpr (D == Dim3) {
        Ip_i[2] = m_particles.i3_prev(p);
        xp_i[2] = static_cast<real_t>(m_particles.i3_prev(p))
                  + static_cast<real_t>(m_particles.dx3_prev(p));
      }

      for (auto i { 0 }; i < static_cast<short>(D); ++i) {
        const real_t xi_mid = HALF * (xp_i[i] + xp_f[i]);
        xp_r[i]
          = math::fmin(static_cast<real_t>(math::fmin(Ip_i[i], Ip_f[i]) + 1),
                       math::fmax(static_cast<real_t>(math::fmax(Ip_i[i], Ip_f[i])), xi_mid));
      }
    }
  };

  template <>
  Inline void DepositCurrents_kernel<Dim2>::depositCurrentsFromParticle(
    const real_t&             weight,
    const vec_t<Dim3>&        vp,
    const tuple_t<int, Dim2>& Ip_f,
    const tuple_t<int, Dim2>& Ip_i,
    const coord_t<Dim2>&      xp_f,
    const coord_t<Dim2>&      xp_i,
    const coord_t<Dim2>&      xp_r) const {
    real_t Wx1_1 { HALF * (xp_i[0] + xp_r[0]) - static_cast<real_t>(Ip_i[0]) };
    real_t Wx1_2 { HALF * (xp_f[0] + xp_r[0]) - static_cast<real_t>(Ip_f[0]) };
    real_t Fx1_1 { (xp_r[0] - xp_i[0]) * weight * m_charge / m_dt };
    real_t Fx1_2 { (xp_f[0] - xp_r[0]) * weight * m_charge / m_dt };

    real_t Wx2_1 { HALF * (xp_i[1] + xp_r[1]) - static_cast<real_t>(Ip_i[1]) };
    real_t Wx2_2 { HALF * (xp_f[1] + xp_r[1]) - static_cast<real_t>(Ip_f[1]) };
    real_t Fx2_1 { (xp_r[1] - xp_i[1]) * weight * m_charge / m_dt };
    real_t Fx2_2 { (xp_f[1] - xp_r[1]) * weight * m_charge / m_dt };

    real_t Fx3_1 { HALF * vp[2] * weight * m_charge };
    real_t Fx3_2 { HALF * vp[2] * weight * m_charge };

    auto   cur_access = m_scatter_cur0.access();
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
    const real_t& weight,
    const vec_t<Dim3>&,
    const tuple_t<int, Dim3>& Ip_f,
    const tuple_t<int, Dim3>& Ip_i,
    const coord_t<Dim3>&      xp_f,
    const coord_t<Dim3>&      xp_i,
    const coord_t<Dim3>&      xp_r) const {
    real_t Wx1_1 { HALF * (xp_i[0] + xp_r[0]) - static_cast<real_t>(Ip_i[0]) };
    real_t Wx1_2 { HALF * (xp_f[0] + xp_r[0]) - static_cast<real_t>(Ip_f[0]) };
    real_t Fx1_1 { (xp_r[0] - xp_i[0]) * weight * m_charge / m_dt };
    real_t Fx1_2 { (xp_f[0] - xp_r[0]) * weight * m_charge / m_dt };

    real_t Wx2_1 { HALF * (xp_i[1] + xp_r[1]) - static_cast<real_t>(Ip_i[1]) };
    real_t Wx2_2 { HALF * (xp_f[1] + xp_r[1]) - static_cast<real_t>(Ip_f[1]) };
    real_t Fx2_1 { (xp_r[1] - xp_i[1]) * weight * m_charge / m_dt };
    real_t Fx2_2 { (xp_f[1] - xp_r[1]) * weight * m_charge / m_dt };

    real_t Wx3_1 { HALF * (xp_i[2] + xp_r[2]) - static_cast<real_t>(Ip_i[2]) };
    real_t Wx3_2 { HALF * (xp_f[2] + xp_r[2]) - static_cast<real_t>(Ip_f[2]) };
    real_t Fx3_1 { (xp_r[2] - xp_i[2]) * weight * m_charge / m_dt };
    real_t Fx3_2 { (xp_f[2] - xp_r[2]) * weight * m_charge / m_dt };

    auto   cur_access = m_scatter_cur0.access();
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

#endif    // GRPIC_CURRENTS_DEPOSIT_H
