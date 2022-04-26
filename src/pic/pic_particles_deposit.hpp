#ifndef PIC_PARTICLES_DEPOSIT_H
#define PIC_PARTICLES_DEPOSIT_H

#include "global.h"
#include "fields.h"
#include "particles.h"
#include "meshblock.h"
#include "pic.h"

#include <stdexcept>

namespace ntt {

  /**
   * Algorithm for the current deposition.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Deposit {
    using index_t = const std::size_t;
    Meshblock<D, SimulationType::PIC> m_mblock;
    Particles<D, SimulationType::PIC> m_particles;
    real_t                            m_coeff, m_dt;

  public:
    Deposit(const Meshblock<D, SimulationType::PIC>& mblock,
            const Particles<D, SimulationType::PIC>& particles,
            const real_t&                            coeff,
            const real_t&                            dt)
      : m_mblock(mblock), m_particles(particles), m_coeff(coeff), m_dt(dt) {}

    /**
     * Loop over all active particles and deposit currents.
     * TODO: forward/backward
     */
    void depositCurrents() {
      auto range_policy = Kokkos::RangePolicy<AccelExeSpace>(0, m_particles.npart());
      Kokkos::parallel_for("deposit", range_policy, *this);
    }

    /*
     * Get particle position in `coord_t` form.
     *
     * @param[in] p Index of particle.
     * @param[out] Ip_f Final position of the particle (cell index).
     * @param[out] dIp_f Final position of the particle (displacement).
     * @param[out] Ip_i Initial position of the particle (cell index).
     * @param[out] dIp_i Initial position of the particle (displacement).
     *
     */
    Inline void getDepositInterval(
      const index_t&, tuple_t<int, D>&, tuple_t<float, D>&, tuple_t<int, D>&, tuple_t<float, D>&) const;

    Inline void operator()(const index_t p) const {
      // _f = final, _i = initial
      tuple_t<int, D>   Ip_f, Ip_i;
      tuple_t<float, D> dIp_f, dIp_i;

      // get [i, di]_init and [i, di]_final (per dimension)
      getDepositInterval(p, Ip_f, dIp_f, Ip_i, dIp_i);
    }
  };

  template <>
  Inline void Deposit<Dimension::ONE_D>::getDepositInterval(const index_t&                    p,
                                                            tuple_t<int, Dimension::ONE_D>&   Ip_f,
                                                            tuple_t<float, Dimension::ONE_D>& dIp_f,
                                                            tuple_t<int, Dimension::ONE_D>&   Ip_i,
                                                            tuple_t<float, Dimension::ONE_D>& dIp_i) const {
    coord_t<Dimension::ONE_D> xp;
    vec_t<Dimension::THREE_D> vp;
    real_t                    inv_energy;

    Ip_f[0]  = m_particles.i1(p);
    dIp_f[0] = m_particles.dx1(p);

    xp[0] = static_cast<real_t>(Ip_f[0]) + static_cast<real_t>(dIp_f[0]);

    m_mblock.metric.v_Cart2Cntrv(xp, {m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p)}, vp);

    inv_energy = SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p)) + SQR(m_particles.ux3(p));
    inv_energy = ONE / math::sqrt(ONE + inv_energy);

    // get particle 3-velocity in coordinate basis
    for (short i {0}; i < 3; ++i) {
      vp[i] *= inv_energy;
    }

    xp[0] -= m_dt * vp[0];
    auto [I1_i, dI1_i] = m_mblock.metric.CU_to_Idi(xp[0]);
    Ip_i[0]            = I1_i;
    dIp_i[0]           = dI1_i;
  }
  template <>
  Inline void Deposit<Dimension::TWO_D>::getDepositInterval(const index_t&                    p,
                                                            tuple_t<int, Dimension::TWO_D>&   Ip_f,
                                                            tuple_t<float, Dimension::TWO_D>& dIp_f,
                                                            tuple_t<int, Dimension::TWO_D>&   Ip_i,
                                                            tuple_t<float, Dimension::TWO_D>& dIp_i) const {
    coord_t<Dimension::TWO_D> xp;
    vec_t<Dimension::THREE_D> vp;
    real_t                    inv_energy;

    Ip_f[0]  = m_particles.i1(p);
    dIp_f[0] = m_particles.dx1(p);

    Ip_f[1]  = m_particles.i2(p);
    dIp_f[1] = m_particles.dx2(p);

    xp[0] = static_cast<real_t>(Ip_f[0]) + static_cast<real_t>(dIp_f[0]);
    xp[1] = static_cast<real_t>(Ip_f[1]) + static_cast<real_t>(dIp_f[1]);

    m_mblock.metric.v_Cart2Cntrv(xp, {m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p)}, vp);

    inv_energy = SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p)) + SQR(m_particles.ux3(p));
    inv_energy = ONE / math::sqrt(ONE + inv_energy);

    // get particle 3-velocity in coordinate basis
    for (short i {0}; i < 3; ++i) {
      vp[i] *= inv_energy;
    }

    xp[0] -= m_dt * vp[0];
    xp[1] -= m_dt * vp[1];
    auto [I1_i, dI1_i] = m_mblock.metric.CU_to_Idi(xp[0]);
    auto [I2_i, dI2_i] = m_mblock.metric.CU_to_Idi(xp[1]);
    Ip_i[0]            = I1_i;
    dIp_i[0]           = dI1_i;
    Ip_i[1]            = I2_i;
    dIp_i[1]           = dI2_i;
  }
  template <>
  Inline void Deposit<Dimension::THREE_D>::getDepositInterval(const index_t&                      p,
                                                              tuple_t<int, Dimension::THREE_D>&   Ip_f,
                                                              tuple_t<float, Dimension::THREE_D>& dIp_f,
                                                              tuple_t<int, Dimension::THREE_D>&   Ip_i,
                                                              tuple_t<float, Dimension::THREE_D>& dIp_i) const {
    coord_t<Dimension::THREE_D> xp;
    vec_t<Dimension::THREE_D>   vp;
    real_t                      inv_energy;

    Ip_f[0]  = m_particles.i1(p);
    dIp_f[0] = m_particles.dx1(p);
    Ip_f[1]  = m_particles.i2(p);
    dIp_f[1] = m_particles.dx2(p);
    Ip_f[2]  = m_particles.i3(p);
    dIp_f[2] = m_particles.dx3(p);

    xp[0] = static_cast<real_t>(Ip_f[0]) + static_cast<real_t>(dIp_f[0]);
    xp[1] = static_cast<real_t>(Ip_f[1]) + static_cast<real_t>(dIp_f[1]);
    xp[2] = static_cast<real_t>(Ip_f[2]) + static_cast<real_t>(dIp_f[2]);
    m_mblock.metric.v_Cart2Cntrv(xp, {m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p)}, vp);

    inv_energy = SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p)) + SQR(m_particles.ux3(p));
    inv_energy = ONE / math::sqrt(ONE + inv_energy);

    // get particle 3-velocity in coordinate basis
    for (short i {0}; i < 3; ++i) {
      vp[i] *= inv_energy;
    }

    xp[0] -= m_dt * vp[0];
    xp[1] -= m_dt * vp[1];
    xp[2] -= m_dt * vp[2];
    auto [I1_i, dI1_i] = m_mblock.metric.CU_to_Idi(xp[0]);
    auto [I2_i, dI2_i] = m_mblock.metric.CU_to_Idi(xp[1]);
    auto [I3_i, dI3_i] = m_mblock.metric.CU_to_Idi(xp[2]);
    Ip_i[0]            = I1_i;
    dIp_i[0]           = dI1_i;
    Ip_i[1]            = I2_i;
    dIp_i[1]           = dI2_i;
    Ip_i[2]            = I3_i;
    dIp_i[2]           = dI3_i;
  }

} // namespace ntt

#endif
