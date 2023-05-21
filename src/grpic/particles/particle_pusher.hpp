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
     * @brief Iterative geodesic pusher substep for momentum only.
     * @tparam T Push type (Photon_t or Massive_t)
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param vp_upd updated particle velocity [return].
     */
    template <typename T>
    Inline void GeodesicMomentumPush(T,
                                     const coord_t<D>&  xp,
                                     const vec_t<Dim3>& vp,
                                     vec_t<Dim3>&       vp_upd) const {}

    /**
     * @brief Iterative geodesic pusher substep for coordinate only.
     * @tparam T Push type (Photon_t or Massive_t)
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param xp_upd updated particle coordinate [return].
     */
    template <typename T>
    Inline void GeodesicCoordinatePush(T,
                                       const coord_t<D>&  xp,
                                       const vec_t<Dim3>& vp,
                                       coord_t<D>&        xp_upd) const {}

    /**
     * @brief Iterative geodesic pusher substep (old method).
     * @tparam T Push type (Photon_t or Massive_t)
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param xp_upd updated particle coordinate [return].
     * @param vp_upd updated particle velocity [return].
     */
    template <typename T>
    Inline void GeodesicFullPush(T,
                                 const coord_t<D>&  xp,
                                 const vec_t<Dim3>& vp,
                                 coord_t<D>&        xp_upd,
                                 vec_t<Dim3>&       vp_upd) const {}

    /**
     * @brief Iterative geodesic pusher substep (old method).
     * @tparam T Push type (Photon_t or Massive_t)
     * @param xp particle coordinate (at n + 1/2 for leapfrog, at n for old scheme).
     * @param vp particle velocity (at n + 1/2 for leapfrog, at n for old scheme).
     * @param phi updated phi [return].
     */
    template <typename T>
    Inline void UpdatePhi(T, const coord_t<D>& xp, const vec_t<Dim3>& vp, real_t& phi) const {}

    /**
     * @brief EM pusher (Boris) substep.
     * @param xp coordinate of the particle.
     * @param vp covariant velocity of the particle.
     * @param Dp_hat hatted electric field at the particle position.
     * @param Bp_hat hatted magnetic field at the particle position.
     * @param v_upd updated covarient velocity of the particle [return].
     */
    Inline void EMHalfPush(const coord_t<D>&  xp,
                           const vec_t<Dim3>& vp,
                           vec_t<Dim3>&       Dp_hat,
                           vec_t<Dim3>&       Bp_hat,
                           vec_t<Dim3>&       vp_upd) const {
      vec_t<Dim3> vp_hat { ZERO }, vp_upd_hat { ZERO };
      m_mblock.metric.v3_Cov2Hat(xp, vp, vp_upd_hat);

      // this is a half-push
      real_t COEFF { m_coeff * HALF * m_mblock.metric.alpha(xp) };

      Dp_hat[0] *= COEFF;
      Dp_hat[1] *= COEFF;
      Dp_hat[2] *= COEFF;

      vp_upd_hat[0] += Dp_hat[0];
      vp_upd_hat[1] += Dp_hat[1];
      vp_upd_hat[2] += Dp_hat[2];

      COEFF
        *= ONE
           / math::sqrt(ONE + SQR(vp_upd_hat[0]) + SQR(vp_upd_hat[1]) + SQR(vp_upd_hat[2]));
      Bp_hat[0] *= COEFF;
      Bp_hat[1] *= COEFF;
      Bp_hat[2] *= COEFF;
      COEFF = TWO / (ONE + SQR(Bp_hat[0]) + SQR(Bp_hat[1]) + SQR(Bp_hat[2]));

      vp_hat[0]
        = (vp_upd_hat[0] + vp_upd_hat[1] * Bp_hat[2] - vp_upd_hat[2] * Bp_hat[1]) * COEFF;
      vp_hat[1]
        = (vp_upd_hat[1] + vp_upd_hat[2] * Bp_hat[0] - vp_upd_hat[0] * Bp_hat[2]) * COEFF;
      vp_hat[2]
        = (vp_upd_hat[2] + vp_upd_hat[0] * Bp_hat[1] - vp_upd_hat[1] * Bp_hat[0]) * COEFF;

      vp_upd_hat[0] += vp_hat[1] * Bp_hat[2] - vp_hat[2] * Bp_hat[1] + Dp_hat[0];
      vp_upd_hat[1] += vp_hat[2] * Bp_hat[0] - vp_hat[0] * Bp_hat[2] + Dp_hat[1];
      vp_upd_hat[2] += vp_hat[0] * Bp_hat[1] - vp_hat[1] * Bp_hat[0] + Dp_hat[2];

      m_mblock.metric.v3_Hat2Cov(xp, vp_upd_hat, vp_upd);
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

    /**
     * @brief Compute the gamma parameter Gamma = sqrt(u_i u_j h^ij) for massless particles.
     */
    Inline auto computeGamma(const Photon_t&,
                             const vec_t<Dim3>& u_cov,
                             const vec_t<Dim3>& u_cntrv) const -> real_t {
      return math::sqrt(u_cov[0] * u_cntrv[0] + u_cov[1] * u_cntrv[1] + u_cov[2] * u_cntrv[2]);
    }

    /**
     * @brief Compute the gamma parameter Gamma = sqrt(1 + u_i u_j h^ij) for massive particles.
     */
    Inline auto computeGamma(const Massive_t&,
                             const vec_t<Dim3>& u_cov,
                             const vec_t<Dim3>& u_cntrv) const -> real_t {
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
    inline constexpr int    N_ITER { 10 };
  }    // namespace

#define DERIVATIVE_IN_R(func, x)                                                              \
  (HALF_OVR_EPSILON                                                                           \
   * (m_mblock.metric.func({ x[0] + EPSILON, x[1] })                                          \
      - m_mblock.metric.func({ x[0] - EPSILON, x[1] })))

#define DERIVATIVE_IN_TH(func, x)                                                             \
  (HALF_OVR_EPSILON                                                                           \
   * (m_mblock.metric.func({ x[0], x[1] + EPSILON })                                          \
      - m_mblock.metric.func({ x[0], x[1] - EPSILON })))

  template <>
  template <typename T>
  Inline void Pusher_kernel<Dim2>::GeodesicMomentumPush(T,
                                                        const coord_t<Dim2>& xp,
                                                        const vec_t<Dim3>&   vp,
                                                        vec_t<Dim3>&         vp_upd) const {
    // initialize midpoint values & updated values
    vec_t<Dim3> vp_mid { ZERO };
    vec_t<Dim3> vp_mid_cntrv { ZERO };

#pragma unroll
    for (int i = 0; i < N_ITER; i++) {
      // find midpoint values
      vp_mid[0] = HALF * (vp[0] + vp_upd[0]);
      vp_mid[1] = HALF * (vp[1] + vp_upd[1]);
      vp_mid[2] = vp[2];

      // find contravariant midpoint velocity
      m_mblock.metric.v3_Cov2Cntrv(xp, vp_mid, vp_mid_cntrv);

      // find Gamma / alpha at midpoint
      real_t u0 { computeGamma(T {}, vp_mid, vp_mid_cntrv) / m_mblock.metric.alpha(xp) };

      // find updated velocity
      vp_upd[0] = vp[0]
                  + m_dt
                      * (-m_mblock.metric.alpha(xp) * u0 * DERIVATIVE_IN_R(alpha, xp)
                         + vp_mid[0] * DERIVATIVE_IN_R(beta1, xp)
                         - (HALF / u0)
                             * (DERIVATIVE_IN_R(h11, xp) * SQR(vp_mid[0])
                                + DERIVATIVE_IN_R(h22, xp) * SQR(vp_mid[1])
                                + DERIVATIVE_IN_R(h33, xp) * SQR(vp_mid[2])
                                + TWO * DERIVATIVE_IN_R(h13, xp) * vp_mid[0] * vp_mid[2]));
      vp_upd[1] = vp[1]
                  + m_dt
                      * (-m_mblock.metric.alpha(xp) * u0 * DERIVATIVE_IN_TH(alpha, xp)
                         + vp_mid[1] * DERIVATIVE_IN_TH(beta1, xp)
                         - (HALF / u0)
                             * (DERIVATIVE_IN_TH(h11, xp) * SQR(vp_mid[0])
                                + DERIVATIVE_IN_TH(h22, xp) * SQR(vp_mid[1])
                                + DERIVATIVE_IN_TH(h33, xp) * SQR(vp_mid[2])
                                + TWO * DERIVATIVE_IN_TH(h13, xp) * vp_mid[0] * vp_mid[2]));
    }
  }

  template <>
  template <typename T>
  Inline void Pusher_kernel<Dim2>::GeodesicCoordinatePush(T,
                                                          const coord_t<Dim2>& xp,
                                                          const vec_t<Dim3>&   vp,
                                                          coord_t<Dim2>&       xp_upd) const {
    vec_t<Dim3>   vp_cntrv { ZERO };
    coord_t<Dim2> xp_mid { ZERO };

#pragma unroll
    for (int i = 0; i < N_ITER; i++) {
      // find midpoint values
      xp_mid[0] = HALF * (xp[0] + xp_upd[0]);
      xp_mid[1] = HALF * (xp[1] + xp_upd[1]);

      // find contravariant midpoint velocity
      m_mblock.metric.v3_Cov2Cntrv(xp_mid, vp, vp_cntrv);
      real_t gamma = computeGamma(T {}, vp, vp_cntrv);

      // find midpoint coefficients
      real_t u0 { gamma / m_mblock.metric.alpha(xp_mid) };

      // find updated coordinate shift
      xp_upd[0] = xp[0] + m_dt * (vp_cntrv[0] / u0 - m_mblock.metric.beta1(xp_mid));
      xp_upd[1] = xp[1] + m_dt * (vp_cntrv[1] / u0);
    }
  }

  template <>
  template <typename T>
  Inline void Pusher_kernel<Dim2>::GeodesicFullPush(T,
                                                    const coord_t<Dim2>& xp,
                                                    const vec_t<Dim3>&   vp,
                                                    coord_t<Dim2>&       xp_upd,
                                                    vec_t<Dim3>&         vp_upd) const {
    // initialize midpoint values & updated values
    vec_t<Dim2> xp_mid { ZERO };
    vec_t<Dim3> vp_mid { ZERO }, vp_mid_cntrv { ZERO };

#pragma unroll
    for (int i = 0; i < N_ITER; i++) {
      xp_mid[0] = HALF * (xp[0] + xp_upd[0]);
      xp_mid[1] = HALF * (xp[1] + xp_upd[1]);

      // find midpoint values
      vp_mid[0] = HALF * (vp[0] + vp_upd[0]);
      vp_mid[1] = HALF * (vp[1] + vp_upd[1]);
      vp_mid[2] = vp[2];

      // find contravariant midpoint velocity
      m_mblock.metric.v3_Cov2Cntrv(xp_mid, vp_mid, vp_mid_cntrv);

      // find Gamma / alpha at midpoint
      real_t u0 { computeGamma(T {}, vp_mid, vp_mid_cntrv) / m_mblock.metric.alpha(xp_mid) };

      // find updated coordinate shift
      xp_upd[0] = xp[0] + m_dt * (vp_mid_cntrv[0] / u0 - m_mblock.metric.beta1(xp_mid));
      xp_upd[1] = xp[1] + m_dt * (vp_mid_cntrv[1] / u0);

      // find updated velocity
      vp_upd[0] = vp[0]
                  + m_dt
                      * (-m_mblock.metric.alpha(xp_mid) * u0 * DERIVATIVE_IN_R(alpha, xp_mid)
                         + vp_mid[0] * DERIVATIVE_IN_R(beta1, xp_mid)
                         - (HALF / u0)
                             * (DERIVATIVE_IN_R(h11, xp_mid) * SQR(vp_mid[0])
                                + DERIVATIVE_IN_R(h22, xp_mid) * SQR(vp_mid[1])
                                + DERIVATIVE_IN_R(h33, xp_mid) * SQR(vp_mid[2])
                                + TWO * DERIVATIVE_IN_R(h13, xp_mid) * vp_mid[0] * vp_mid[2]));
      vp_upd[1]
        = vp[1]
          + m_dt
              * (-m_mblock.metric.alpha(xp_mid) * u0 * DERIVATIVE_IN_TH(alpha, xp_mid)
                 + vp_mid[1] * DERIVATIVE_IN_TH(beta1, xp_mid)
                 - (HALF / u0)
                     * (DERIVATIVE_IN_TH(h11, xp_mid) * SQR(vp_mid[0])
                        + DERIVATIVE_IN_TH(h22, xp_mid) * SQR(vp_mid[1])
                        + DERIVATIVE_IN_TH(h33, xp_mid) * SQR(vp_mid[2])
                        + TWO * DERIVATIVE_IN_TH(h13, xp_mid) * vp_mid[0] * vp_mid[2]));
    }
  }

#undef DERIVATIVE_IN_TH
#undef DERIVATIVE_IN_R

  /* -------------------------------------------------------------------------- */
  /*                                 Phi pusher                                 */
  /* -------------------------------------------------------------------------- */
  template <>
  template <typename T>
  Inline void Pusher_kernel<Dim2>::UpdatePhi(T,
                                             const coord_t<Dim2>& xp,
                                             const vec_t<Dim3>&   vp,
                                             real_t&              phi) const {
    vec_t<Dim3> vp_cntrv { ZERO };
    m_mblock.metric.v3_Cov2Cntrv(xp, vp, vp_cntrv);
    real_t u0 { computeGamma(T {}, vp, vp_cntrv) / m_mblock.metric.alpha(xp) };
    phi += m_dt * vp_cntrv[2] / u0;
    if (phi >= constant::TWO_PI) {
      phi -= constant::TWO_PI;
    } else if (phi < ZERO) {
      phi += constant::TWO_PI;
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

    // first order interpolation
    // Using fields em::e = D(t = n), and em::b0 = B(t = n)

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

    // B0x1
    b0[0] = ((HALF * (B0X1(i, j) + B0X1(i, j - 1))) * (ONE - dx1)
             + (HALF * (B0X1(i + 1, j) + B0X1(i + 1, j - 1))) * dx1)
              * (ONE - dx2)
            + ((HALF * (B0X1(i, j) + B0X1(i, j + 1))) * (ONE - dx1)
               + (HALF * (B0X1(i + 1, j) + B0X1(i + 1, j + 1))) * dx1)
                * dx2;
    // B0x2
    b0[1] = ((HALF * (B0X2(i - 1, j) + B0X2(i, j))) * (ONE - dx1)
             + (HALF * (B0X2(i, j) + B0X2(i + 1, j))) * dx1)
              * (ONE - dx2)
            + ((HALF * (B0X2(i - 1, j + 1) + B0X2(i, j + 1))) * (ONE - dx1)
               + (HALF * (B0X2(i, j + 1) + B0X2(i + 1, j + 1))) * dx1)
                * dx2;
    // B0x3
    b0[2]
      = ((INV_4 * (B0X3(i - 1, j - 1) + B0X3(i - 1, j) + B0X3(i, j - 1) + B0X3(i, j)))
           * (ONE - dx1)
         + (INV_4 * (B0X3(i, j - 1) + B0X3(i, j) + B0X3(i + 1, j - 1) + B0X3(i + 1, j))) * dx1)
          * (ONE - dx2)
        + ((INV_4 * (B0X3(i - 1, j) + B0X3(i - 1, j + 1) + B0X3(i, j) + B0X3(i, j + 1)))
             * (ONE - dx1)
           + (INV_4 * (B0X3(i, j) + B0X3(i, j + 1) + B0X3(i + 1, j) + B0X3(i + 1, j + 1)))
               * dx1)
            * dx2;
  }

  /* ------------------------------ Photon pusher ----------------------------- */
  template <>
  Inline void Pusher_kernel<Dim2>::operator()(Photon_t, index_t p) const {
    if (m_particles.tag(p) == static_cast<short>(ParticleTag::alive)) {
      // record previous coordinate
      m_particles.i1_prev(p)  = m_particles.i1(p);
      m_particles.i2_prev(p)  = m_particles.i2(p);
      m_particles.dx1_prev(p) = m_particles.dx1(p);
      m_particles.dx2_prev(p) = m_particles.dx2(p);

      coord_t<Dim2> xp { ZERO };
      vec_t<Dim3>   vp { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) };

      xp[0] = get_prtl_x1(m_particles, p);
      xp[1] = get_prtl_x2(m_particles, p);

      coord_t<Dim2> xp_upd { xp[0], xp[1] };
      vec_t<Dim3>   vp_upd { vp[0], vp[1], vp[2] };

      /* ----------------------------- Leapfrog pusher ---------------------------- */
      // u_i(n - 1/2) -> u_i(n + 1/2)
      GeodesicMomentumPush<Photon_t>(Photon_t {}, xp, vp, vp_upd);
      // x^i(n) -> x^i(n + 1)
      GeodesicCoordinatePush<Photon_t>(Photon_t {}, xp, vp_upd, xp_upd);
      // update phi
      UpdatePhi<Photon_t>(Photon_t {},
                          { (xp[0] + xp_upd[0]) * HALF, (xp[1] + xp_upd[1]) * HALF },
                          vp_upd,
                          m_particles.phi(p));

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

  /* ------------------------- Massive particle pusher ------------------------ */

  template <>
  Inline void Pusher_kernel<Dim2>::operator()(Massive_t, index_t p) const {
    if (m_particles.tag(p) == static_cast<short>(ParticleTag::alive)) {
      // record previous coordinate
      m_particles.i1_prev(p)  = m_particles.i1(p);
      m_particles.i2_prev(p)  = m_particles.i2(p);
      m_particles.dx1_prev(p) = m_particles.dx1(p);
      m_particles.dx2_prev(p) = m_particles.dx2(p);

      coord_t<Dim2> xp { ZERO };

      xp[0] = get_prtl_x1(m_particles, p);
      xp[1] = get_prtl_x2(m_particles, p);

      coord_t<Dim2> xp_upd { xp[0], xp[1] };

      // vec_t<Dim3> Dp_cntrv { ZERO }, Bp_cntrv { ZERO }, Dp_hat { ZERO }, Bp_hat { ZERO };
      // interpolateFields(p, Dp_cntrv, Bp_cntrv);
      // m_mblock.metric.v3_Cntrv2Hat(xp, Dp_cntrv, Dp_hat);
      // m_mblock.metric.v3_Cntrv2Hat(xp, Bp_cntrv, Bp_hat);

      vec_t<Dim3>   vp { m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p) };
      vec_t<Dim3>   vp_upd { vp[0], vp[1], vp[2] };

      /* -------------------------------- Leapfrog -------------------------------- */
      /* u_i(n - 1/2) -> u*_i(n - 1/2) */
      // EMHalfPush(xp, vp, Dp_hat, Bp_hat, vp_upd);
      // vp[0] = vp_upd[0];
      // vp[1] = vp_upd[1];
      // vp[2] = vp_upd[2];
      /* u*_i(n - 1/2) -> u*_i(n + 1/2) */
      GeodesicMomentumPush<Massive_t>(Massive_t {}, xp, vp, vp_upd);
      // vp[0] = vp_upd[0];
      // vp[1] = vp_upd[1];
      // vp[2] = vp_upd[2];
      /* u*_i(n + 1/2) -> u_i(n + 1/2) */
      // EMHalfPush(xp, vp, Dp_hat, Bp_hat, vp_upd);
      /* x^i(n) -> x^i(n + 1) */
      GeodesicCoordinatePush<Massive_t>(Massive_t {}, xp, vp_upd, xp_upd);

      // update phi
      UpdatePhi<Massive_t>(Massive_t {},
                           { (xp[0] + xp_upd[0]) * HALF, (xp[1] + xp_upd[1]) * HALF },
                           vp_upd,
                           m_particles.phi(p));

      /* ------------------------------- Old pusher ------------------------------- */
      // GeodesicFullPush<Massive_t>(Massive_t {}, xp, vp, xp_upd, vp_upd);
      // UpdatePhi<Massive_t>(Massive_t {}, { xp_upd[0], xp_upd[1] }, vp_upd,
      // m_particles.phi(p));

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

}    // namespace ntt

#endif