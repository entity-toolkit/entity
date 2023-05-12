#ifndef GRPIC_PARTICLE_PUSHER_H
#define GRPIC_PARTICLE_PUSHER_H

#include "wrapper.h"

#include "field_macros.h"
#include "fields.h"
#include "grpic.h"
#include "meshblock.h"
#include "particle_macros.h"
#include "particles.h"
#include "qmath.h"

#include <iostream>
#include <stdexcept>

namespace ntt {
  struct Massive_t {};
  struct Photon_t {};

  namespace {
    inline constexpr real_t EPSILON { 1e-6 };
    inline constexpr real_t HALF_OVR_EPSILON { HALF / EPSILON };
  }    // namespace

  /**
   * @brief Algorithm for the Particle pusher.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Pusher_kernel {
    Meshblock<D, GRPICEngine> m_mblock;
    Particles<D, GRPICEngine> m_particles;
    const real_t              m_coeff, m_dt;
    const int                 m_ni2;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param particles Particles.
     * @param coeff Coefficient to be multiplied by dE/dt = coeff * curl B.
     * @param dt Time step.
     */
    Pusher_kernel(const Meshblock<D, GRPICEngine>& mblock,
                  const Particles<D, GRPICEngine>& particles,
                  const real_t&                    coeff,
                  const real_t&                    dt)
      : m_mblock(mblock),
        m_particles(particles),
        m_coeff(coeff),
        m_dt(dt),
        m_ni2 { mblock.Ni2() } {}
    /**
     * @brief Loop over all active particles of the given species and call the appropriate
     * pusher.
     */
    void apply() {
      if (m_particles.pusher() == ParticlePusher::PHOTON) {
        // push photons
        auto range_policy
          = Kokkos::RangePolicy<AccelExeSpace, Photon_t>(0, m_particles.npart());
        Kokkos::parallel_for("pusher", range_policy, *this);
      } else if (m_particles.pusher() == ParticlePusher::BORIS) {
        // // push boris-particles
        // auto range_policy
        //   = Kokkos::RangePolicy<AccelExeSpace, Boris_t>(0, m_particles.npart());
        // Kokkos::parallel_for("pusher", range_policy, *this);
      } else if (m_particles.pusher() == ParticlePusher::NONE) {
        // do nothing
      } else {
        NTTHostError("not implemented");
      }
    }

    /**
     * @brief Pusher for the forward Boris algorithm.
     * @param p index.
     */
    template <typename T>
    Inline void operator()(const T&, index_t p) const {}

    // /**
    //  * @brief Pusher for the photon.
    //  * @param p index.
    //  */
    // Inline void operator()(const Photon_t&, index_t p) const {}

    /**
     * @brief Transform particle coordinate from code units i+di to `real_t` type.
     * @param p index of the particle.
     * @param coord coordinate of the particle as a vector (of size D).
     */
    Inline void getParticleCoordinate(index_t&, coord_t<D>&) const;

    /**
     * @brief First order Yee mesh field interpolation to particle position.
     * @param p index of the particle.
     * @param e interpolated e-field vector of size 3 [return].
     * @param b interpolated b-field vector of size 3 [return].
     */
    Inline void interpolateFields(index_t&, vec_t<Dim3>&, vec_t<Dim3>&) const;

    /**
     * @brief Update particle positions according to updated velocities.
     * @param p index of the particle.
     * @param v particle 3-velocity.
     */
    Inline void positionUpdate(index_t&, const vec_t<Dim3>&) const;

    /**
     * @brief Update each position component.
     * @param p index of the particle.
     * @param v corresponding 3-velocity component.
     */
    Inline void positionUpdate_x1(index_t&, const real_t&) const;
    Inline void positionUpdate_x2(index_t&, const real_t&) const;
    Inline void positionUpdate_x3(index_t&, const real_t&) const;

    /**
     * @brief Boris algorithm.
     * @note Fields are modified inside the function and cannot be reused.
     * @param p index of the particle.
     * @param e interpolated e-field vector of size 3 [modified].
     * @param b interpolated b-field vector of size 3 [modified].
     */
    Inline void BorisUpdate(index_t&, vec_t<Dim3>&, vec_t<Dim3>&) const;

    Inline auto ComputeEnergy(const Photon_t&, vec_t<Dim3>& u) const -> real_t {
      return math::sqrt(SQR(u[0]) + SQR(u[1]) + SQR(u[2]));
    }

    Inline auto ComputeEnergy(const Massive_t&, vec_t<Dim3>& u) const -> real_t {
      return math::sqrt(ONE + SQR(u[0]) + SQR(u[1]) + SQR(u[2]));
    }
  };

  template <>
  Inline void Pusher_kernel<Dim2>::getParticleCoordinate(index_t& p, coord_t<Dim2>& xp) const {
    xp[0] = get_prtl_x1(m_particles, p);
    xp[1] = get_prtl_x2(m_particles, p);
  }
  template <>
  Inline void Pusher_kernel<Dim3>::getParticleCoordinate(index_t& p, coord_t<Dim3>& xp) const {
    xp[0] = get_prtl_x1(m_particles, p);
    xp[1] = get_prtl_x2(m_particles, p);
    xp[2] = get_prtl_x3(m_particles, p);
  }

  template <>
  template <typename T>
  Inline void Pusher_kernel<Dim2>::operator()(const T&, index_t p) const {
    if (m_particles.tag(p) == static_cast<short>(ParticleTag::alive)) {
      coord_t<Dim2> xp { ZERO };
      getParticleCoordinate(p, xp);
      vec_t<Dim3>   vp { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) };

      // initialize midpoint values & updated values
      coord_t<Dim2> xp_mid { ZERO }, xp_upd { xp[0], xp[1] };
      vec_t<Dim3>   vp_mid { ZERO };
      vec_t<Dim3>   vp_mid_upd { ZERO };
      vec_t<Dim3>   vp_upd { vp[0], vp[1], vp[2] };

      coord_t<Dim2> xp_dr_P { xp[0] + EPSILON, xp[1] };
      coord_t<Dim2> xp_dr_M { xp[0] - EPSILON, xp[1] };
      coord_t<Dim2> xp_dth_P { xp[0], xp[1] + EPSILON };
      coord_t<Dim2> xp_dth_M { xp[0], xp[1] - EPSILON };

      // iterate
#pragma unroll
      for (int i = 0; i < 10; i++) {
        // find midpoint values
        xp_mid[0] = HALF * (xp[0] + xp_upd[0]);
        xp_mid[1] = HALF * (xp[1] + xp_upd[1]);
        vp_mid[0] = HALF * (vp[0] + vp_upd[0]);
        vp_mid[1] = HALF * (vp[1] + vp_upd[1]);
        vp_mid[2] = HALF * (vp[2] + vp_upd[2]);

        // find contravariant midpoint velocity
        m_mblock.metric.v_Cov2Cntrv(xp_mid, vp_mid, vp_mid_upd);

        // find spacial derivatives
        real_t dalpha_dr {
          HALF_OVR_EPSILON * (m_mblock.metric.alpha(xp_dr_P) - m_mblock.metric.alpha(xp_dr_M))
        };
        real_t dbeta_dr { HALF_OVR_EPSILON
                          * (m_mblock.metric.beta1u(xp_dr_P)
                             - m_mblock.metric.beta1u(xp_dr_M)) };

        real_t dh11_dr { HALF_OVR_EPSILON
                         * (m_mblock.metric.h_11_inv(xp_dr_P)
                            - m_mblock.metric.h_11_inv(xp_dr_M)) };
        real_t dh22_dr { HALF_OVR_EPSILON
                         * (m_mblock.metric.h_22_inv(xp_dr_P)
                            - m_mblock.metric.h_22_inv(xp_dr_M)) };
        real_t dh33_dr { HALF_OVR_EPSILON
                         * (m_mblock.metric.h_33_inv(xp_dr_P)
                            - m_mblock.metric.h_33_inv(xp_dr_M)) };

        real_t dh33_dth { HALF_OVR_EPSILON
                          * (m_mblock.metric.h_33_inv(xp_dth_P)
                             - m_mblock.metric.h_33_inv(xp_dth_M)) };

        // find midpoint coefficients
        real_t u0 { ComputeEnergy(T {}, vp_mid) / m_mblock.metric.alpha(xp_mid) };

        // find updated coordinate shift
        xp_upd[0] = xp[0] + m_dt * (vp_mid_upd[0] / u0 - m_mblock.metric.beta1u(xp_mid));
        xp_upd[1] = xp[1] + m_dt * (vp_mid_upd[1] / u0);

        // find updated velocity
        vp_upd[0]
          = vp[0]
            + m_dt
                * (-m_mblock.metric.alpha(xp_mid) * u0 * dalpha_dr + vp_mid[0] * dbeta_dr
                   - (HALF / u0)
                       * (dh11_dr * SQR(vp_mid[0]) + dh22_dr * SQR(vp_mid[1])
                          + dh33_dr * SQR(vp_mid[2])));
        vp_upd[1] = vp[1] - m_dt * (HALF / u0) * (dh33_dth * SQR(vp_mid[2]));
        vp_upd[2] = vp[2];
      }

      // update coordinate
      int   i1, i2;
      float dx1, dx2;
      from_Xi_to_i_di(xp_upd[0], i1, dx1);
      from_Xi_to_i_di(xp_upd[1], i2, dx2);
      m_particles.i1(p)  = i1;
      m_particles.dx1(p) = dx1;
      m_particles.i2(p)  = i2;
      m_particles.dx2(p) = dx2;

      // update velocity
      m_particles.ux1(p) = vp_upd[0];
      m_particles.ux2(p) = vp_upd[1];
      m_particles.ux3(p) = vp_upd[2];
    }
  }

  template <>
  Inline void Pusher_kernel<Dim2>::interpolateFields(index_t&     p,
                                                     vec_t<Dim3>& e0,
                                                     vec_t<Dim3>& b0) const {
    const auto   i { m_particles.i1(p) + N_GHOSTS };
    const real_t dx1 { static_cast<real_t>(m_particles.dx1(p)) };
    const auto   j { m_particles.i2(p) + N_GHOSTS };
    const real_t dx2 { static_cast<real_t>(m_particles.dx2(p)) };

    // first order
    real_t       c000, c100, c010, c110, c00, c10;

    // Ex1
    // interpolate to nodes
    c000  = HALF * (EX1(i, j) + EX1(i - 1, j));
    c100  = HALF * (EX1(i, j) + EX1(i + 1, j));
    c010  = HALF * (EX1(i, j + 1) + EX1(i - 1, j + 1));
    c110  = HALF * (EX1(i, j + 1) + EX1(i + 1, j + 1));
    // interpolate from nodes to the particle position
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    e0[0] = c00 * (ONE - dx2) + c10 * dx2;
    // Ex2
    c000  = HALF * (EX2(i, j) + EX2(i, j - 1));
    c100  = HALF * (EX2(i + 1, j) + EX2(i + 1, j - 1));
    c010  = HALF * (EX2(i, j) + EX2(i, j + 1));
    c110  = HALF * (EX2(i + 1, j) + EX2(i + 1, j + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    e0[1] = c00 * (ONE - dx2) + c10 * dx2;
    // Ex3
    c000  = EX3(i, j);
    c100  = EX3(i + 1, j);
    c010  = EX3(i, j + 1);
    c110  = EX3(i + 1, j + 1);
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    e0[2] = c00 * (ONE - dx2) + c10 * dx2;

    // Bx1
    c000  = HALF * (BX1(i, j) + BX1(i, j - 1));
    c100  = HALF * (BX1(i + 1, j) + BX1(i + 1, j - 1));
    c010  = HALF * (BX1(i, j) + BX1(i, j + 1));
    c110  = HALF * (BX1(i + 1, j) + BX1(i + 1, j + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    b0[0] = c00 * (ONE - dx2) + c10 * dx2;
    // Bx2
    c000  = HALF * (BX2(i - 1, j) + BX2(i, j));
    c100  = HALF * (BX2(i, j) + BX2(i + 1, j));
    c010  = HALF * (BX2(i - 1, j + 1) + BX2(i, j + 1));
    c110  = HALF * (BX2(i, j + 1) + BX2(i + 1, j + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    b0[1] = c00 * (ONE - dx2) + c10 * dx2;
    // Bx3
    c000  = INV_4 * (BX3(i - 1, j - 1) + BX3(i - 1, j) + BX3(i, j - 1) + BX3(i, j));
    c100  = INV_4 * (BX3(i, j - 1) + BX3(i, j) + BX3(i + 1, j - 1) + BX3(i + 1, j));
    c010  = INV_4 * (BX3(i - 1, j) + BX3(i - 1, j + 1) + BX3(i, j) + BX3(i, j + 1));
    c110  = INV_4 * (BX3(i, j) + BX3(i, j + 1) + BX3(i + 1, j) + BX3(i + 1, j + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    b0[2] = c00 * (ONE - dx2) + c10 * dx2;
  }

  template <>
  Inline void Pusher_kernel<Dim3>::interpolateFields(index_t&,
                                                     vec_t<Dim3>&,
                                                     vec_t<Dim3>&) const {
    NTTError("not implemented");
  }

}    // namespace ntt

#endif