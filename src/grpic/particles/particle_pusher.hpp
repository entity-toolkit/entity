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

#include <stdio.h>

#include <iostream>
#include <stdexcept>

namespace ntt {
  struct Massive_t {};
  struct Photon_t {};

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
        // push massive particles
        auto range_policy
          = Kokkos::RangePolicy<AccelExeSpace, Massive_t>(0, m_particles.npart());
        Kokkos::parallel_for("pusher", range_policy, *this);
      } else if (m_particles.pusher() == ParticlePusher::NONE) {
        // do nothing
      } else {
        NTTHostError("not implemented");
      }
    }

    /**
     * @brief Main pusher subroutine for photon particles.
     */
    Inline void operator()(Photon_t, index_t p) const {}

    /**
     * @brief Main pusher subroutine for massive particles.
     */
    Inline void operator()(Massive_t, index_t p) const {}

    /**
     * @brief Iterative geodesic pusher substep.
     * @tparam T Push type (Photon_t or Massive_t)
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param xp_upd updated particle coordinate [return].
     * @param vp_upd updated particle velocity [return].
     */
    template <typename T>
    Inline void GeodesicPush(T,
                             const coord_t<D>&  xp,
                             const vec_t<Dim3>& vp,
                             coord_t<D>&        xp_upd,
                             vec_t<Dim3>&       vp_upd) const {}

    /**
     * @brief Iterative geodesic pusher substep for coordinate only (massives).
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param xp_upd updated particle coordinate [return].
     */
    Inline void GeodesicCoordinatePush(const coord_t<D>&  xp,
                                       const vec_t<Dim3>& vp,
                                       coord_t<D>&        xp_upd) const {}

    /**
     * @brief EM pusher substep.
     * @param xp coordinate of the particle.
     * @param vp covariant velocity of the particle.
     * @param Dp_hat hatted electric field at the particle position.
     * @param Bp_hat hatted magnetic field at the particle position.
     * @param v_upd updated covarient velocity of the particle [return].
     */
    Inline void EMPush(const coord_t<D>&  xp,
                       const vec_t<Dim3>& vp,
                       const vec_t<Dim3>& Dp_hat,
                       const vec_t<Dim3>& Bp_hat,
                       vec_t<Dim3>&       vp_upd) const {
      vec_t<Dim3> vp_hat { ZERO }, vp_upd_hat { ZERO };
      m_mblock.metric.v_Cov2Hat(xp, vp, vp_hat);

      // !ASK: is this correct?
      vp_hat[0] += Dp_hat[0];
      vp_hat[1] += Dp_hat[1];
      vp_hat[2] += Dp_hat[2];

      const real_t inv_gamma {
        ONE / math::sqrt(ONE + SQR(vp_hat[0]) + SQR(vp_hat[1]) + SQR(vp_hat[2]))
      };
      const real_t prefactor { m_dt / TWO * m_mblock.metric.alpha(xp) };

      vec_t<Dim3>  tt { prefactor * inv_gamma * Bp_hat[0],
                       prefactor * inv_gamma * Bp_hat[1],
                       prefactor * inv_gamma * Bp_hat[2] };
      const real_t ff { ONE / math::sqrt(ONE + SQR(tt[0]) + SQR(tt[1]) + SQR(tt[2])) };

      vp_upd_hat[0] = ff * (vp_hat[0] + vp_hat[1] * tt[2] - vp_hat[2] * tt[1]);
      vp_upd_hat[1] = ff * (vp_hat[1] + vp_hat[2] * tt[0] - vp_hat[0] * tt[2]);
      vp_upd_hat[2] = ff * (vp_hat[2] + vp_hat[0] * tt[1] - vp_hat[1] * tt[0]);

      m_mblock.metric.v_Hat2Cov(xp, vp_upd_hat, vp_upd);
    }

    // Helper functions

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
    Inline void interpolateFields(index_t& p, vec_t<Dim3>& e, vec_t<Dim3>& b) const {};

    // /**
    //  * @brief Boris algorithm.
    //  * @note Fields are modified inside the function and cannot be reused.
    //  * @param p index of the particle.
    //  * @param e interpolated e-field vector of size 3 [modified].
    //  * @param b interpolated b-field vector of size 3 [modified].
    //  */
    // Inline void BorisUpdate(index_t&, vec_t<Dim3>&, vec_t<Dim3>&) const;

    Inline auto ComputeEnergy(const Photon_t&, vec_t<Dim3>& u_cov, vec_t<Dim3>& u_cntrv) const
      -> real_t {
      return math::sqrt(u_cov[0] * u_cntrv[0] + u_cov[1] * u_cntrv[1] + u_cov[2] * u_cntrv[2]);
    }

    Inline auto ComputeEnergy(const Massive_t&, vec_t<Dim3>& u_cov, vec_t<Dim3>& u_cntrv) const
      -> real_t {
      return math::sqrt(ONE + u_cov[0] * u_cntrv[0] + u_cov[1] * u_cntrv[1]
                        + u_cov[2] * u_cntrv[2]);
    }
  };

  /* -------------------------------------------------------------------------- */
  /*                               Geodesic pusher                              */
  /* -------------------------------------------------------------------------- */
  namespace {
    inline constexpr real_t EPSILON { 1e-2 };
    inline constexpr real_t HALF_OVR_EPSILON { HALF / EPSILON };
  }    // namespace

#define DERIVATIVE_IN_R(func)                                                                 \
  (HALF_OVR_EPSILON                                                                           \
   * (m_mblock.metric.func({ xp_mid[0] + EPSILON, xp_mid[1] })                                \
      - m_mblock.metric.func({ xp_mid[0] - EPSILON, xp_mid[1] })))

#define DERIVATIVE_IN_TH(func)                                                                \
  (HALF_OVR_EPSILON                                                                           \
   * (m_mblock.metric.func({ xp_mid[0], xp_mid[1] + EPSILON })                                \
      - m_mblock.metric.func({ xp_mid[0], xp_mid[1] - EPSILON })))

#define ATMIDPOINT(func) (m_mblock.metric.func(xp_mid))

  template <>
  template <typename T>
  Inline void Pusher_kernel<Dim2>::GeodesicPush(T,
                                                const coord_t<Dim2>& xp,
                                                const vec_t<Dim3>&   vp,
                                                coord_t<Dim2>&       xp_upd,
                                                vec_t<Dim3>&         vp_upd) const {
    // initialize midpoint values & updated values
    coord_t<Dim2> xp_mid { ZERO };
    vec_t<Dim3>   vp_mid { ZERO };
    vec_t<Dim3>   vp_mid_cntrv { ZERO };

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
      m_mblock.metric.v_Cov2Cntrv(xp_mid, vp_mid, vp_mid_cntrv);

      // find Gamma / alpha at midpoint
      real_t u0 { ComputeEnergy(T {}, vp_mid, vp_mid_cntrv) / ATMIDPOINT(alpha) };

      // find updated coordinate shift
      xp_upd[0] = xp[0] + m_dt * (vp_mid_cntrv[0] / u0 - ATMIDPOINT(beta1));
      xp_upd[1] = xp[1] + m_dt * (vp_mid_cntrv[1] / u0);

      // find updated velocity
      vp_upd[0] = vp[0]
                  + m_dt
                      * (-ATMIDPOINT(alpha) * u0 * DERIVATIVE_IN_R(alpha)
                         + vp_mid[0] * DERIVATIVE_IN_R(beta1)
                         - (HALF / u0)
                             * (DERIVATIVE_IN_R(h11) * SQR(vp_mid[0])
                                + DERIVATIVE_IN_R(h22) * SQR(vp_mid[1])
                                + DERIVATIVE_IN_R(h33) * SQR(vp_mid[2])
                                + TWO * DERIVATIVE_IN_R(h13) * vp_mid[0] * vp_mid[2]));
      vp_upd[1] = vp[1]
                  + m_dt
                      * (-ATMIDPOINT(alpha) * u0 * DERIVATIVE_IN_TH(alpha)
                         + vp_mid[1] * DERIVATIVE_IN_TH(beta1)
                         - (HALF / u0)
                             * (DERIVATIVE_IN_TH(h11) * SQR(vp_mid[0])
                                + DERIVATIVE_IN_TH(h22) * SQR(vp_mid[1])
                                + DERIVATIVE_IN_TH(h33) * SQR(vp_mid[2])
                                + TWO * DERIVATIVE_IN_TH(h13) * vp_mid[0] * vp_mid[2]));
      vp_upd[2] = vp[2];
    }
  }

  template <>
  Inline void Pusher_kernel<Dim2>::GeodesicCoordinatePush(const coord_t<Dim2>& xp,
                                                          const vec_t<Dim3>&   vp,
                                                          coord_t<Dim2>&       xp_upd) const {
    vec_t<Dim3>   vp_cntrv { ZERO };
    // vec_t<Dimension::THREE_D> v { vx1, vx2, vx3 };
    // vec_t<Dimension::THREE_D> vu { v[0], v[1], v[2] };
    // real_t                    gamma { math::sqrt(v[0] * vu[0] + v[1] * vu[1] + v[2] * vu[2])
    // };

    // initialize midpoint values & updated values
    coord_t<Dim2> xp_mid {ZERO};

    // iterate
    for (int i = 0; i < 10; i++) {
      // find midpoint values
      xp_mid[0] = HALF * (xp[0] + xp_upd[0]);
      xp_mid[1] = HALF * (xp[1] + xp_upd[1]);

      // find contravariant midpoint velocity
      m_mblock.metric.v_Cov2Cntrv(xp_mid, vp, vp_cntrv);

      // find midpoint coefficients
      real_t u0 { gamma / m_mblock.metric.alpha(xp_mid) };

      // find updated coordinate shift
      xp_upd[0] = xp[0] + m_dt * (vp_cntrv[0] / u0 - m_mblock.metric.beta1(xp_mid));
      xp_upd[1] = xp[1] + m_dt * (vp_cntrv[1] / u0);
    }
  }

#undef ATMIDPOINT
#undef DERIVATIVE_IN_TH
#undef DERIVATIVE_IN_R

  template <>
  Inline void Pusher_kernel<Dim2>::interpolateFields(index_t&     p,
                                                     vec_t<Dim3>& e0,
                                                     vec_t<Dim3>& b0) const {
    const auto   i { m_particles.i1(p) + N_GHOSTS };
    const real_t dx1 { static_cast<real_t>(m_particles.dx1(p)) };
    const auto   j { m_particles.i2(p) + N_GHOSTS };
    const real_t dx2 { static_cast<real_t>(m_particles.dx2(p)) };

    // first order interpolation

    // Ex1
    e0[0] = ((HALF * (EX1(i, j) + EX1(i - 1, j))) * (ONE - dx1)
             + (HALF * (EX1(i, j) + EX1(i + 1, j))) * dx1)
              * (ONE - dx2)
            + ((HALF * (EX1(i, j + 1) + EX1(i - 1, j + 1))) * (ONE - dx1)
               + (HALF * (EX1(i, j + 1) + EX1(i + 1, j + 1))) * dx1)
                * dx2;
    // Ex2
    e0[1] = ((HALF * (EX2(i, j) + EX2(i, j - 1))) * (ONE - dx1)
             + (HALF * (EX2(i + 1, j) + EX2(i + 1, j - 1))) * dx1)
              * (ONE - dx2)
            + ((HALF * (EX2(i, j) + EX2(i, j + 1))) * (ONE - dx1)
               + (HALF * (EX2(i + 1, j) + EX2(i + 1, j + 1))) * dx1)
                * dx2;
    // Ex3
    e0[2] = ((EX3(i, j)) * (ONE - dx1) + (EX3(i + 1, j)) * dx1) * (ONE - dx2)
            + ((EX3(i, j + 1)) * (ONE - dx1) + (EX3(i + 1, j + 1)) * dx1) * dx2;

    // Bx1
    b0[0] = ((HALF * (BX1(i, j) + BX1(i, j - 1))) * (ONE - dx1)
             + (HALF * (BX1(i + 1, j) + BX1(i + 1, j - 1))) * dx1)
              * (ONE - dx2)
            + ((HALF * (BX1(i, j) + BX1(i, j + 1))) * (ONE - dx1)
               + (HALF * (BX1(i + 1, j) + BX1(i + 1, j + 1))) * dx1)
                * dx2;
    // Bx2
    b0[1] = ((HALF * (BX2(i - 1, j) + BX2(i, j))) * (ONE - dx1)
             + (HALF * (BX2(i, j) + BX2(i + 1, j))) * dx1)
              * (ONE - dx2)
            + ((HALF * (BX2(i - 1, j + 1) + BX2(i, j + 1))) * (ONE - dx1)
               + (HALF * (BX2(i, j + 1) + BX2(i + 1, j + 1))) * dx1)
                * dx2;
    // Bx3
    b0[2]
      = ((INV_4 * (BX3(i - 1, j - 1) + BX3(i - 1, j) + BX3(i, j - 1) + BX3(i, j))) * (ONE - dx1)
         + (INV_4 * (BX3(i, j - 1) + BX3(i, j) + BX3(i + 1, j - 1) + BX3(i + 1, j))) * dx1)
          * (ONE - dx2)
        + ((INV_4 * (BX3(i - 1, j) + BX3(i - 1, j + 1) + BX3(i, j) + BX3(i, j + 1)))
             * (ONE - dx1)
           + (INV_4 * (BX3(i, j) + BX3(i, j + 1) + BX3(i + 1, j) + BX3(i + 1, j + 1))) * dx1)
            * dx2;
  }

  /* ------------------------------ Photon pusher ----------------------------- */
  template <>
  Inline void Pusher_kernel<Dim2>::operator()(Photon_t, index_t p) const {
    if (m_particles.tag(p) == static_cast<short>(ParticleTag::alive)) {
      coord_t<Dim2> xp { ZERO };
      vec_t<Dim3>   vp { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) };

      xp[0] = get_prtl_x1(m_particles, p);
      xp[1] = get_prtl_x2(m_particles, p);

      coord_t<Dim2> xp_upd { xp[0], xp[1] };
      vec_t<Dim3>   vp_upd { vp[0], vp[1], vp[2] };

      GeodesicPush<Photon_t>(Photon_t {}, xp, vp, xp_upd, vp_upd);

      // update coordinate
      int   i1, i2;
      float dx1, dx2;
      from_Xi_to_i_di(xp_upd[0], i1, dx1);
      from_Xi_to_i_di(xp_upd[1], i2, dx2);
      m_particles.i1(p)  = i1;
      m_particles.dx1(p) = dx1;
      m_particles.i2(p)  = i2;
      m_particles.dx2(p) = dx2;

      // update phi
      // vp used to store contravariant velocity
      m_mblock.metric.v_Cov2Cntrv(xp_upd, vp_upd, vp);
      real_t u0 { ComputeEnergy(Photon_t {}, vp_upd, vp) / m_mblock.metric.alpha(xp_upd) };
      m_particles.phi(p) += m_dt * vp[2] / u0;

      // update velocity
      m_particles.ux1(p) = vp_upd[0];
      m_particles.ux2(p) = vp_upd[1];
      m_particles.ux3(p) = vp_upd[2];
    }
  }

  /* ------------------------- Massive particle pusher ------------------------ */

  // coord_t<D> xp;
  // getParticleCoordinate(p, xp);

  // vec_t<Dim3> Dp_cntrv, Bp_cntrv, Dp_hat, Bp_hat;
  // interpolateFields(p, Dp_cntrv, Bp_cntrv);
  // m_mblock.metric.v_Cntrv2Hat(xp, Dp_cntrv, Dp_hat);
  // m_mblock.metric.v_Cntrv2Hat(xp, Bp_cntrv, Bp_hat);

  // BorisUpdate(p, Dp_hat, Bp_hat);
  // velocityUpdate(p, m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p));
  // BorisUpdate(p, Dp_hat, Bp_hat);
  // coordinateUpdate(p, m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p));

  // GeodesicPush<Massive_t>(Massive_t {}, p);
  // }

  template <>
  Inline void Pusher_kernel<Dim2>::operator()(Massive_t, index_t p) const {
    if (m_particles.tag(p) == static_cast<short>(ParticleTag::alive)) {
      coord_t<Dim2> xp { ZERO }, xp_upd { ZERO };

      xp[0] = get_prtl_x1(m_particles, p);
      xp[1] = get_prtl_x2(m_particles, p);

      vec_t<Dim3> Dp_cntrv { ZERO }, Bp_cntrv { ZERO }, Dp_hat { ZERO }, Bp_hat { ZERO };
      interpolateFields(p, Dp_cntrv, Bp_cntrv);
      m_mblock.metric.v_Cntrv2Hat(xp, Dp_cntrv, Dp_hat);
      m_mblock.metric.v_Cntrv2Hat(xp, Bp_cntrv, Bp_hat);

      vec_t<Dim3> vp { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) };
      vec_t<Dim3> vp_upd { ZERO };

      // xp: old particle coordinate
      // vp: particle velocity
      // vp_upd = vp
      EMPush(xp, vp, Dp_hat, Bp_hat, vp_upd);
      // vp_upd: updated particle velocity

      // xp: old particle coordinate
      // vp: updated particle velocity
      // xp_upd = xp
      // vp_upd = vp
      xp_upd[0] = xp[0];
      xp_upd[1] = xp[1];
      vp[0]     = vp_upd[0];
      vp[1]     = vp_upd[1];
      vp[2]     = vp_upd[2];
      // only the updated velocity matters after this step
      GeodesicPush<Massive_t>(Massive_t {}, xp, vp, xp_upd, vp_upd);
      vp[0]     = vp_upd[0];
      vp[1]     = vp_upd[1];
      vp[2]     = vp_upd[2];
      xp_upd[0] = xp[0];
      xp_upd[1] = xp[1];
      // vp_upd: updated particle velocity

      // xp: old particle coordinate
      // vp: updated particle velocity
      // xp_upd = xp
      // vp_upd = vp
      EMPush(xp, vp, Dp_hat, Bp_hat, vp_upd);
      vp[0] = vp_upd[0];
      vp[1] = vp_upd[1];
      vp[2] = vp_upd[2];
      // vp_upd: updated particle velocity

      // xp: old particle coordinate
      // vp: updated particle velocity
      // xp_upd = xp
      // vp_upd = vp

      // GeodesicPush<Massive_t>(Massive_t {}, xp, vp, xp_upd, vp_upd);

      // Inline void EMPush(const coord_t<D>&  xp,
      //              const vec_t<Dim3>& vp,
      //              const vec_t<Dim3>& Dp_hat,
      //              const vec_t<Dim3>& Bp_hat,
      //              vec_t<Dim3>&       vp_upd) const {

      // vec_t<Dim3>   vp { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) };

      // xp[0] = get_prtl_x1(m_particles, p);
      // xp[1] = get_prtl_x2(m_particles, p);

      // coord_t<Dim2> xp_upd { xp[0], xp[1] };
      // vec_t<Dim3>   vp_upd { vp[0], vp[1], vp[2] };

      // GeodesicPush<Photon_t>(Photon_t {}, p, xp, vp, xp_upd, vp_upd);

      // // update coordinate
      // int   i1, i2;
      // float dx1, dx2;
      // from_Xi_to_i_di(xp_upd[0], i1, dx1);
      // from_Xi_to_i_di(xp_upd[1], i2, dx2);
      // m_particles.i1(p)  = i1;
      // m_particles.dx1(p) = dx1;
      // m_particles.i2(p)  = i2;
      // m_particles.dx2(p) = dx2;
      // // update phi

      // // vp used to store contravariant velocity
      // m_mblock.metric.v_Cov2Cntrv(xp_upd, vp_upd, vp);
      // real_t u0 { ComputeEnergy(Photon_t {}, vp_upd, vp) / m_mblock.metric.alpha(xp_upd) };
      // m_particles.phi(p) += m_dt * vp[2] / u0;

      // // update velocity
      // m_particles.ux1(p) = vp_upd[0];
      // m_particles.ux2(p) = vp_upd[1];
      // m_particles.ux3(p) = vp_upd[2];
    }
  }

}    // namespace ntt

#endif