#ifndef PIC_CURRENTS_DEPOSIT_H
#define PIC_CURRENTS_DEPOSIT_H

#include "global.h"
#include "fields.h"
#include "particles.h"
#include "meshblock.h"
#include "pic.h"

#include <stdexcept>

namespace ntt {

  /**
   * @brief Algorithm for the current deposition.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Deposit {
    Meshblock<D, SimulationType::PIC> m_mblock;
    Particles<D, SimulationType::PIC> m_particles;
    RealScatterFieldND<D, 3>          m_scatter_cur;
    real_t                            m_charge, m_dt;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param particles Particles.
     * @param scatter_cur Scatter array of the currents.
     * @param charge charge of the species (code units).
     * @param dt Time step.
     */
    Deposit(const Meshblock<D, SimulationType::PIC>& mblock,
            const Particles<D, SimulationType::PIC>& particles,
            const RealScatterFieldND<D, 3>&          scatter_cur,
            const real_t&                            charge,
            const real_t&                            dt)
      : m_mblock(mblock),
        m_particles(particles),
        m_scatter_cur(scatter_cur),
        m_charge(charge),
        m_dt(dt) {}

    /**
     * @brief Loop over all active particles and deposit currents.
     * TODO: forward/backward
     */
    void depositCurrents() {
      auto range_policy = Kokkos::RangePolicy<AccelExeSpace>(0, m_particles.npart());
      Kokkos::parallel_for("deposit", range_policy, *this);
    }

    /**
     * @brief Iteration of the loop over particles.
     * @param p index.
     */
    Inline void operator()(index_t p) const {
      // _f = final, _i = initial
      tuple_t<int, D>           Ip_f, Ip_i;
      coord_t<D>                xp_f, xp_i, xp_r;
      vec_t<Dimension::THREE_D> vp;

      // get [i, di]_init and [i, di]_final (per dimension)
      getDepositInterval(p, vp, Ip_f, Ip_i, xp_f, xp_i, xp_r);
      depositCurrentsFromParticle(vp, Ip_f, Ip_i, xp_f, xp_i, xp_r);
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
    Inline void depositCurrentsFromParticle(const vec_t<Dimension::THREE_D>& vp,
                                            const tuple_t<int, D>&           Ip_f,
                                            const tuple_t<int, D>&           Ip_i,
                                            const coord_t<D>&                xp_f,
                                            const coord_t<D>&                xp_i,
                                            const coord_t<D>&                xp_r) const;

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
    Inline void getDepositInterval(index_t&                   p,
                                   vec_t<Dimension::THREE_D>& vp,
                                   tuple_t<int, D>&           Ip_f,
                                   tuple_t<int, D>&           Ip_i,
                                   coord_t<D>&                xp_f,
                                   coord_t<D>&                xp_i,
                                   coord_t<D>&                xp_r) const {
      coord_t<D>        xmid;
      real_t            inv_energy;
      tuple_t<float, D> dIp_f;

      if constexpr ((D == Dimension::ONE_D) || (D == Dimension::TWO_D)
                    || (D == Dimension::THREE_D)) {
        Ip_f[0]  = m_particles.i1(p);
        dIp_f[0] = m_particles.dx1(p);
      }

      if constexpr ((D == Dimension::TWO_D) || (D == Dimension::THREE_D)) {
        Ip_f[1]  = m_particles.i2(p);
        dIp_f[1] = m_particles.dx2(p);
      }

      if constexpr (D == Dimension::THREE_D) {
        Ip_f[2]  = m_particles.i3(p);
        dIp_f[2] = m_particles.dx3(p);
      }

      for (short i {0}; i < static_cast<short>(D); ++i) {
        xp_f[i] = static_cast<real_t>(Ip_f[i]) + static_cast<real_t>(dIp_f[i]);
      }

      m_mblock.metric.v_Cart2Cntrv(
        xp_f, {m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p)}, vp);

      inv_energy = SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p)) + SQR(m_particles.ux3(p));
      inv_energy = ONE / math::sqrt(ONE + inv_energy);

      // get particle 3-velocity in coordinate basis
      for (short i {0}; i < 3; ++i) {
        vp[i] *= inv_energy;
      }

      for (short i {0}; i < static_cast<short>(D); ++i) {
        xp_i[i]       = xp_f[i] - m_dt * vp[i];
        xmid[i]       = HALF * (xp_i[i] + xp_f[i]);
        auto [I_i, _] = m_mblock.metric.CU_to_Idi(xp_i[i]);
        Ip_i[i]       = I_i;
        xp_r[i]
          = math::fmin(static_cast<real_t>(math::fmin(Ip_i[i], Ip_f[i]) + 1),
                       math::fmax(static_cast<real_t>(math::fmax(Ip_i[i], Ip_f[i])), xmid[i]));
      }
    }
  };

  template <>
  Inline void Deposit<Dimension::ONE_D>::depositCurrentsFromParticle(
    const vec_t<Dimension::THREE_D>&,
    const tuple_t<int, Dimension::ONE_D>&,
    const tuple_t<int, Dimension::ONE_D>&,
    const coord_t<Dimension::ONE_D>&,
    const coord_t<Dimension::ONE_D>&,
    const coord_t<Dimension::ONE_D>&) const {}

  /**
   * !TODO: fix the conversion to I+di
   */
  template <>
  Inline void Deposit<Dimension::TWO_D>::depositCurrentsFromParticle(
    const vec_t<Dimension::THREE_D>&      vp,
    const tuple_t<int, Dimension::TWO_D>& Ip_f,
    const tuple_t<int, Dimension::TWO_D>& Ip_i,
    const coord_t<Dimension::TWO_D>&      xp_f,
    const coord_t<Dimension::TWO_D>&      xp_i,
    const coord_t<Dimension::TWO_D>&      xp_r) const {
    real_t Wx1_1 {HALF * (xp_i[0] + xp_r[0]) - static_cast<real_t>(Ip_i[0])};
    real_t Wx1_2 {HALF * (xp_f[0] + xp_r[0]) - static_cast<real_t>(Ip_f[0])};
    real_t Fx1_1 {(xp_r[0] - xp_i[0]) * m_charge / m_dt};
    real_t Fx1_2 {(xp_f[0] - xp_r[0]) * m_charge / m_dt};

    real_t Wx2_1 {HALF * (xp_i[1] + xp_r[1]) - static_cast<real_t>(Ip_i[1])};
    real_t Wx2_2 {HALF * (xp_f[1] + xp_r[1]) - static_cast<real_t>(Ip_f[1])};
    real_t Fx2_1 {(xp_r[1] - xp_i[1]) * m_charge / m_dt};
    real_t Fx2_2 {(xp_f[1] - xp_r[1]) * m_charge / m_dt};

    real_t Fx3_1 {HALF * vp[2] * m_charge};
    real_t Fx3_2 {HALF * vp[2] * m_charge};

    auto cur_access = m_scatter_cur.access();
    cur_access(Ip_i[0] + N_GHOSTS, Ip_i[1] + N_GHOSTS, cur::jx1) += Fx1_1 * (ONE - Wx2_1);
    cur_access(Ip_i[0] + N_GHOSTS, Ip_i[1] + 1 + N_GHOSTS, cur::jx1) += Fx1_1 * Wx2_1;

    cur_access(Ip_i[0] + N_GHOSTS, Ip_i[1] + N_GHOSTS, cur::jx2) += Fx2_1 * (ONE - Wx1_1);
    cur_access(Ip_i[0] + 1 + N_GHOSTS, Ip_i[1] + N_GHOSTS, cur::jx2) += Fx2_1 * Wx1_1;

    cur_access(Ip_f[0] + N_GHOSTS, Ip_f[1] + N_GHOSTS, cur::jx1) += Fx1_2 * (ONE - Wx2_2);
    cur_access(Ip_f[0] + N_GHOSTS, Ip_f[1] + 1 + N_GHOSTS, cur::jx1) += Fx1_2 * Wx2_2;

    cur_access(Ip_f[0] + N_GHOSTS, Ip_f[1] + N_GHOSTS, cur::jx2) += Fx2_2 * (ONE - Wx1_2);
    cur_access(Ip_f[0] + 1 + N_GHOSTS, Ip_f[1] + N_GHOSTS, cur::jx2) += Fx2_2 * Wx1_2;

    cur_access(Ip_i[0] + N_GHOSTS, Ip_i[1] + N_GHOSTS, cur::jx3)
      += Fx3_1 * (ONE - Wx1_1) * (ONE - Wx2_1);
    cur_access(Ip_i[0] + 1 + N_GHOSTS, Ip_i[1] + N_GHOSTS, cur::jx3)
      += Fx3_1 * Wx1_2 * (ONE - Wx2_1);
    cur_access(Ip_i[0] + N_GHOSTS, Ip_i[1] + 1 + N_GHOSTS, cur::jx3)
      += Fx3_1 * (ONE - Wx1_1) * Wx2_1;
    cur_access(Ip_i[0] + 1 + N_GHOSTS, Ip_i[1] + 1 + N_GHOSTS, cur::jx3)
      += Fx3_1 * Wx1_1 * Wx2_1;

    cur_access(Ip_f[0] + N_GHOSTS, Ip_f[1] + N_GHOSTS, cur::jx3)
      += Fx3_2 * (ONE - Wx1_2) * (ONE - Wx2_2);
    cur_access(Ip_f[0] + 1 + N_GHOSTS, Ip_f[1] + N_GHOSTS, cur::jx3)
      += Fx3_2 * Wx1_2 * (ONE - Wx2_2);
    cur_access(Ip_f[0] + N_GHOSTS, Ip_f[1] + 1 + N_GHOSTS, cur::jx3)
      += Fx3_2 * (ONE - Wx1_2) * Wx2_2;
    cur_access(Ip_f[0] + 1 + N_GHOSTS, Ip_f[1] + 1 + N_GHOSTS, cur::jx3)
      += Fx3_2 * Wx1_2 * Wx2_2;
  }

  template <>
  Inline void Deposit<Dimension::THREE_D>::depositCurrentsFromParticle(
    const vec_t<Dimension::THREE_D>&,
    const tuple_t<int, Dimension::THREE_D>&,
    const tuple_t<int, Dimension::THREE_D>&,
    const coord_t<Dimension::THREE_D>&,
    const coord_t<Dimension::THREE_D>&,
    const coord_t<Dimension::THREE_D>&) const {
    NTTError("Deposit::depositCurrentsFromParticle() not implemented for 3D");
  }

} // namespace ntt

// if constexpr (D == Dimension::THREE_D) {
// real_t Wx3_1 {HALF * (xp_i[2] + xp_r[2]) - static_cast<real_t>(Ip_i[2])};
// real_t Wx3_2 {HALF * (xp_f[2] + xp_r[2]) - static_cast<real_t>(Ip_f[2])};
// real_t Fx3_1 {-(xp_r[2] - xp_i[2]) * m_charge};
// real_t Fx3_2 {-(xp_f[2] - xp_r[2]) * m_charge};

// Kokkos::atomic_add(&m_mblock.cur(Ip_i[0], Ip_i[1], cur::jx1), Fx1_1 * (ONE - Wx2_1));
// Kokkos::atomic_add(&m_mblock.cur(Ip_i[0], Ip_i[1] + 1, cur::jx1), Fx1_1 * Wx2_1);

// Kokkos::atomic_add(&m_mblock.cur(Ip_i[0], Ip_i[1], cur::jx2), Fx2_1 * (ONE - Wx1_1));
// Kokkos::atomic_add(&m_mblock.cur(Ip_i[0] + 1, Ip_i[1], cur::jx2), Fx2_1 * Wx1_1);

// Kokkos::atomic_add(&m_mblock.cur(Ip_f[0], Ip_f[1], cur::jx1), Fx1_2 * (ONE - Wx2_2));
// Kokkos::atomic_add(&m_mblock.cur(Ip_f[0], Ip_f[1] + 1, cur::jx1), Fx1_2 * Wx2_2);

// Kokkos::atomic_add(&m_mblock.cur(Ip_f[0], Ip_f[1], cur::jx2), Fx2_2 * (ONE - Wx1_2));
// Kokkos::atomic_add(&m_mblock.cur(Ip_f[0] + 1, Ip_f[1], cur::jx2), Fx2_2 * Wx1_2);

// Kokkos::atomic_add(&m_mblock.cur(Ip_i[0], Ip_i[1], cur::jx3),
//                    Fx3_1 * (ONE - Wx1_1) * (ONE - Wx2_1));
// Kokkos::atomic_add(&m_mblock.cur(Ip_i[0] + 1, Ip_i[1], cur::jx3),
//                    Fx3_1 * Wx1_2 * (ONE - Wx2_1));
// Kokkos::atomic_add(&m_mblock.cur(Ip_i[0], Ip_i[1] + 1, cur::jx3),
//                    Fx3_1 * (ONE - Wx1_1) * Wx2_1);
// Kokkos::atomic_add(&m_mblock.cur(Ip_i[0] + 1, Ip_i[1] + 1, cur::jx3),
//                    Fx3_1 * Wx1_1 * Wx2_1);

// Kokkos::atomic_add(&m_mblock.cur(Ip_f[0], Ip_f[1], cur::jx3),
//                    Fx3_2 * (ONE - Wx1_2) * (ONE - Wx2_2));
// Kokkos::atomic_add(&m_mblock.cur(Ip_f[0] + 1, Ip_f[1], cur::jx3),
//                    Fx3_2 * Wx1_2 * (ONE - Wx2_2));
// Kokkos::atomic_add(&m_mblock.cur(Ip_f[0], Ip_f[1] + 1, cur::jx3),
//                    Fx3_2 * (ONE - Wx1_2) * Wx2_2);
// Kokkos::atomic_add(&m_mblock.cur(Ip_f[0] + 1, Ip_f[1] + 1, cur::jx3),
//                    Fx3_2 * Wx1_2 * Wx2_2);
//}
#endif
