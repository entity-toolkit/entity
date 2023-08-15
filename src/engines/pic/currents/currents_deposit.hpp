#ifndef PIC_CURRENTS_DEPOSIT_H
#define PIC_CURRENTS_DEPOSIT_H

#include "wrapper.h"

#include "field_macros.h"
#include "particle_macros.h"
#include "pic.h"

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
    Meshblock<D, PICEngine> m_mblock;
    Particles<D, PICEngine> m_particles;
    scatter_ndfield_t<D, 3> m_scatter_cur;
    const real_t            m_charge;
    const bool              m_use_weights;
    const real_t            m_dt;
    const int               m_ni2;
    const bool              m_ax_i2min;
    const bool              m_ax_i2max;

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
                           const bool&                    use_weights,
                           const real_t&                  dt)
      : m_mblock(mblock),
        m_particles(particles),
        m_scatter_cur(scatter_cur),
        m_charge(charge),
        m_use_weights { use_weights },
        m_dt(dt),
        m_ni2((int)(m_mblock.Ni2())),
        m_ax_i2min { (mblock.boundaries.size() > 1)
                     && (mblock.boundaries[1][0] == BoundaryCondition::AXIS) },
        m_ax_i2max { (mblock.boundaries.size() > 1)
                     && (mblock.boundaries[1][1] == BoundaryCondition::AXIS) } {}

    /**
     * @brief Iteration of the loop over particles.
     * @param p index.
     */
    Inline auto operator()(index_t p) const -> void {
      if (m_particles.tag(p) != ParticleTag::alive) {
        return;
      }

      // _f = final, _i = initial
      tuple_t<int, D> Ip_f, Ip_i;
      coord_t<D>      xp_f, xp_i, xp_r;
      vec_t<Dim3>     vp { ZERO, ZERO, ZERO };
      const auto      weight = (m_use_weights ? m_particles.weight(p) : ONE);

      // get [i, di]_init and [i, di]_final (per dimension)
      getDepositInterval(p, vp, Ip_f, Ip_i, xp_f, xp_i, xp_r);
      depositCurrentsFromParticle(weight, vp, Ip_f, Ip_i, xp_f, xp_i, xp_r);
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
    Inline auto depositCurrentsFromParticle(const real_t&          weight,
                                            const vec_t<Dim3>&     vp,
                                            const tuple_t<int, D>& Ip_f,
                                            const tuple_t<int, D>& Ip_i,
                                            const coord_t<D>&      xp_f,
                                            const coord_t<D>&      xp_i,
                                            const coord_t<D>&      xp_r) const -> void;

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
    Inline auto getDepositInterval(index_t&         p,
                                   vec_t<Dim3>&     vp,
                                   tuple_t<int, D>& Ip_f,
                                   tuple_t<int, D>& Ip_i,
                                   coord_t<D>&      xp_f,
                                   coord_t<D>&      xp_i,
                                   coord_t<D>&      xp_r) const -> void;
  };

#ifdef MINKOWSKI_METRIC
  // 1D
  template <>
  Inline auto DepositCurrents_kernel<Dim1>::getDepositInterval(index_t&            p,
                                                               vec_t<Dim3>&        vp,
                                                               tuple_t<int, Dim1>& Ip_f,
                                                               tuple_t<int, Dim1>& Ip_i,
                                                               coord_t<Dim1>&      xp_f,
                                                               coord_t<Dim1>&      xp_i,
                                                               coord_t<Dim1>&      xp_r) const
    -> void {
    Ip_f[0] = m_particles.i1(p);
    xp_f[0] = static_cast<real_t>(Ip_f[0]) + static_cast<real_t>(m_particles.dx1(p));

    m_mblock.metric.v3_Cart2Cntrv(
      xp_f, { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) }, vp);
    const real_t inv_energy = ONE / get_prtl_Gamma_SR(m_particles, p);

    vp[0] *= inv_energy;
    vp[1] *= inv_energy;
    vp[2] *= inv_energy;

    xp_i[0] = xp_f[0] - m_dt * vp[0];

    Ip_i[0] = static_cast<int>(xp_i[0]);
    xp_r[0] = math::fmin(static_cast<real_t>(math::fmin(Ip_i[0], Ip_f[0]) + 1),
                         math::fmax(static_cast<real_t>(math::fmax(Ip_i[0], Ip_f[0])),
                                    HALF * (xp_i[0] + xp_f[0])));
  }
  // 2D
  template <>
  Inline auto DepositCurrents_kernel<Dim2>::getDepositInterval(index_t&            p,
                                                               vec_t<Dim3>&        vp,
                                                               tuple_t<int, Dim2>& Ip_f,
                                                               tuple_t<int, Dim2>& Ip_i,
                                                               coord_t<Dim2>&      xp_f,
                                                               coord_t<Dim2>&      xp_i,
                                                               coord_t<Dim2>&      xp_r) const
    -> void {
    Ip_f[0] = m_particles.i1(p);
    Ip_f[1] = m_particles.i2(p);

    xp_f[0] = static_cast<real_t>(Ip_f[0]) + static_cast<real_t>(m_particles.dx1(p));
    xp_f[1] = static_cast<real_t>(Ip_f[1]) + static_cast<real_t>(m_particles.dx2(p));

    m_mblock.metric.v3_Cart2Cntrv(
      xp_f, { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) }, vp);
    const real_t inv_energy = ONE / get_prtl_Gamma_SR(m_particles, p);

    // get particle 3-velocity in coordinate basis
    vp[0] *= inv_energy;
    vp[1] *= inv_energy;
    vp[2] *= inv_energy;

#  pragma unroll
    for (auto i { 0 }; i < 2; ++i) {
      xp_i[i] = xp_f[i] - m_dt * vp[i];
      Ip_i[i] = static_cast<int>(xp_i[i]);
      xp_r[i] = math::fmin(static_cast<real_t>(math::fmin(Ip_i[i], Ip_f[i]) + 1),
                           math::fmax(static_cast<real_t>(math::fmax(Ip_i[i], Ip_f[i])),
                                      HALF * (xp_i[i] + xp_f[i])));
    }
  }
#else     // not MINKOWSKI_METRIC
  template <>
  Inline auto DepositCurrents_kernel<Dim1>::getDepositInterval(index_t&,
                                                               vec_t<Dim3>&,
                                                               tuple_t<int, Dim1>&,
                                                               tuple_t<int, Dim1>&,
                                                               coord_t<Dim1>&,
                                                               coord_t<Dim1>&,
                                                               coord_t<Dim1>&) const -> void {
    NTTError("should not be called");
  }

  template <>
  Inline auto DepositCurrents_kernel<Dim2>::getDepositInterval(index_t&            p,
                                                               vec_t<Dim3>&        vp,
                                                               tuple_t<int, Dim2>& Ip_f,
                                                               tuple_t<int, Dim2>& Ip_i,
                                                               coord_t<Dim2>&      xp_f,
                                                               coord_t<Dim2>&      xp_i,
                                                               coord_t<Dim2>&      xp_r) const
    -> void {
    Ip_f[0] = m_particles.i1(p);
    Ip_f[1] = m_particles.i2(p);

    xp_f[0] = static_cast<real_t>(Ip_f[0]) + static_cast<real_t>(m_particles.dx1(p));
    xp_f[1] = static_cast<real_t>(Ip_f[1]) + static_cast<real_t>(m_particles.dx2(p));

    m_mblock.metric.v3_Cart2Cntrv(
      { xp_f[0], xp_f[1], m_particles.phi(p) },
      { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) },
      vp);

    // make sure the velocity is defined at the axis
    if (m_ax_i2min && Ip_f[1] == 0
        && AlmostEqual(m_particles.dx2(p), static_cast<prtldx_t>(0.0))) {
      vp[2] = ZERO;
    } else if (m_ax_i2max && Ip_f[1] == m_ni2 - 1
               && AlmostEqual(m_particles.dx2(p), static_cast<prtldx_t>(1.0))) {
      vp[2] = ZERO;
    }
    const real_t inv_energy = ONE / get_prtl_Gamma_SR(m_particles, p);

    // get particle 3-velocity in coordinate basis
    vp[0] *= inv_energy;
    vp[1] *= inv_energy;
    vp[2] *= inv_energy;

    xp_i[0] = xp_f[0] - m_dt * vp[0];
    Ip_i[0] = static_cast<int>(xp_i[0]);
    xp_r[0] = math::fmin(static_cast<real_t>(math::fmin(Ip_i[0], Ip_f[0]) + 1),
                         math::fmax(static_cast<real_t>(math::fmax(Ip_i[0], Ip_f[0])),
                                    HALF * (xp_i[0] + xp_f[0])));

    xp_i[1] = xp_f[1] - m_dt * vp[1];
    Ip_i[1] = static_cast<int>(xp_i[1]);
    // reflect off the axis
    if (m_ax_i2min && xp_i[1] < ZERO) {
      Ip_i[1] = 0;
      xp_i[1] = -xp_i[1];
    } else if (m_ax_i2max && xp_i[1] >= static_cast<real_t>(m_ni2)) {
      xp_i[1] = TWO * static_cast<real_t>(m_ni2) - xp_i[1];
      Ip_i[1] = m_ni2 - 1;
    }
    xp_r[1] = math::fmin(static_cast<real_t>(math::fmin(Ip_i[1], Ip_f[1]) + 1),
                         math::fmax(static_cast<real_t>(math::fmax(Ip_i[1], Ip_f[1])),
                                    HALF * (xp_i[1] + xp_f[1])));
  }
#endif    // MINKOWSKI_METRIC

  // 3D
  template <>
  Inline auto DepositCurrents_kernel<Dim3>::getDepositInterval(index_t&            p,
                                                               vec_t<Dim3>&        vp,
                                                               tuple_t<int, Dim3>& Ip_f,
                                                               tuple_t<int, Dim3>& Ip_i,
                                                               coord_t<Dim3>&      xp_f,
                                                               coord_t<Dim3>&      xp_i,
                                                               coord_t<Dim3>&      xp_r) const
    -> void {
    Ip_f[0] = m_particles.i1(p);
    Ip_f[1] = m_particles.i2(p);
    Ip_f[2] = m_particles.i3(p);

    xp_f[0] = static_cast<real_t>(Ip_f[0]) + static_cast<real_t>(m_particles.dx1(p));
    xp_f[1] = static_cast<real_t>(Ip_f[1]) + static_cast<real_t>(m_particles.dx2(p));
    xp_f[2] = static_cast<real_t>(Ip_f[2]) + static_cast<real_t>(m_particles.dx3(p));

    m_mblock.metric.v3_Cart2Cntrv(
      xp_f, { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) }, vp);
    const real_t inv_energy = ONE / get_prtl_Gamma_SR(m_particles, p);

    // get particle 3-velocity in coordinate basis
    vp[0] *= inv_energy;
    vp[1] *= inv_energy;
    vp[2] *= inv_energy;

#pragma unroll
    for (auto i { 0 }; i < 3; ++i) {
      xp_i[i] = xp_f[i] - m_dt * vp[i];
      Ip_i[i] = static_cast<int>(xp_i[i]);
      xp_r[i] = math::fmin(static_cast<real_t>(math::fmin(Ip_i[i], Ip_f[i]) + 1),
                           math::fmax(static_cast<real_t>(math::fmax(Ip_i[i], Ip_f[i])),
                                      HALF * (xp_i[i] + xp_f[i])));
    }
  }

  /**
   * !PERFORM: fix the conversion to I+di
   */
  template <>
  Inline auto DepositCurrents_kernel<Dim1>::depositCurrentsFromParticle(
    const real_t&             weight,
    const vec_t<Dim3>&        vp,
    const tuple_t<int, Dim1>& Ip_f,
    const tuple_t<int, Dim1>& Ip_i,
    const coord_t<Dim1>&      xp_f,
    const coord_t<Dim1>&      xp_i,
    const coord_t<Dim1>&      xp_r) const -> void {
    real_t Wx1_1 { HALF * (xp_i[0] + xp_r[0]) - static_cast<real_t>(Ip_i[0]) };
    real_t Wx1_2 { HALF * (xp_f[0] + xp_r[0]) - static_cast<real_t>(Ip_f[0]) };
    real_t Fx1_1 { (xp_r[0] - xp_i[0]) * weight * m_charge / m_dt };
    real_t Fx1_2 { (xp_f[0] - xp_r[0]) * weight * m_charge / m_dt };

    real_t Fx2_1 { HALF * vp[1] * weight * m_charge };
    real_t Fx2_2 { HALF * vp[1] * weight * m_charge };

    real_t Fx3_1 { HALF * vp[2] * weight * m_charge };
    real_t Fx3_2 { HALF * vp[2] * weight * m_charge };

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

  template <>
  Inline auto DepositCurrents_kernel<Dim2>::depositCurrentsFromParticle(
    const real_t&             weight,
    const vec_t<Dim3>&        vp,
    const tuple_t<int, Dim2>& Ip_f,
    const tuple_t<int, Dim2>& Ip_i,
    const coord_t<Dim2>&      xp_f,
    const coord_t<Dim2>&      xp_i,
    const coord_t<Dim2>&      xp_r) const -> void {
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
  Inline auto DepositCurrents_kernel<Dim3>::depositCurrentsFromParticle(
    const real_t& weight,
    const vec_t<Dim3>&,
    const tuple_t<int, Dim3>& Ip_f,
    const tuple_t<int, Dim3>& Ip_i,
    const coord_t<Dim3>&      xp_f,
    const coord_t<Dim3>&      xp_i,
    const coord_t<Dim3>&      xp_r) const -> void {
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
