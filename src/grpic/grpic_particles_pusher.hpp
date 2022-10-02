#ifndef GRPIC_PARTICLES_PUSHER_H
#define GRPIC_PARTICLES_PUSHER_H

#include "global.h"
#include "fields.h"
#include "particles.h"
#include "meshblock.h"
#include "grpic.h"

#include <stdexcept>

namespace ntt {
  struct Photon_t {};
  struct MassiveFwd_t {};
  /**
   * Algorithm for the Particle pusher.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Pusher {
    using index_t = const std::size_t;
    Meshblock<D, SimulationType::GRPIC> m_mblock;
    Particles<D, SimulationType::GRPIC> m_particles;
    real_t                              m_coeff, m_dt;

  public:
    Pusher(const Meshblock<D, SimulationType::GRPIC>& mblock,
           const Particles<D, SimulationType::GRPIC>& particles,
           const real_t&                              coeff,
           const real_t&                              dt)
      : m_mblock(mblock), m_particles(particles), m_coeff(coeff), m_dt(dt) {}
    /**
     * Loop over all active particles of the given species and call the appropriate pusher.
     * TODO: forward/backward
     */
    void pushParticles() {
      if (m_particles.pusher() == ParticlePusher::PHOTON) {
        // push photons
        auto range_policy
          = Kokkos::RangePolicy<AccelExeSpace, Photon_t>(0, m_particles.npart());
        Kokkos::parallel_for("pusher", range_policy, *this);
      } else if (m_particles.pusher() == ParticlePusher::BORIS) {
        // push boris-particles
        if (SIGN(m_coeff) == SIGN(m_particles.charge())) {
          // push forward
          auto range_policy
            = Kokkos::RangePolicy<AccelExeSpace, MassiveFwd_t>(0, m_particles.npart());
          Kokkos::parallel_for("pusher", range_policy, *this);
        } else {
          //// push backward
          // auto range_policy
          //   = Kokkos::RangePolicy<AccelExeSpace, BorisBwd_t>(0, m_particles.npart());
          // Kokkos::parallel_for("pusher", range_policy, *this);
        }
      } else {
        NTTHostError("pusher not implemented");
      }
    }

    Inline void operator()(const MassiveFwd_t&, const index_t p) const {
      coord_t<D> xp;
      getParticleCoordinate(p, xp);

      vec_t<Dimension::THREE_D> d_int, b_int, d_int_tetrads, b_int_tetrads;
      interpolateFields(p, d_int, b_int);
      m_mblock.metric.v_Cntrv2Hat(xp, d_int, d_int_tetrads);
      m_mblock.metric.v_Cntrv2Hat(xp, b_int, b_int_tetrads);

      BorisUpdate(p, d_int_tetrads, b_int_tetrads);
      velocityUpdate(p, m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p));
      BorisUpdate(p, d_int_tetrads, b_int_tetrads);
      coordinateUpdate(p, m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p));
    }

    /**
     * TODO: Faster sqrt method?
     */
    Inline void operator()(const Photon_t&, const index_t p) const {
      // get coordinate & velocity
      coord_t<D> xp;
      getParticleCoordinate(p, xp);
      vec_t<Dimension::THREE_D> v {m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p)};

      // initialize midpoint values & updated values
      coord_t<D>                xpm;
      coord_t<D>                xpu {xp[0], xp[1]};
      vec_t<Dimension::THREE_D> vm;
      vec_t<Dimension::THREE_D> vmu;
      vec_t<Dimension::THREE_D> vu {v[0], v[1], v[2]};

      // initialize coefficients & coordinates for derivatives
      real_t EPS_R {1e-6};
      real_t EPS_T {1e-6};
      real_t DIVIDE_EPS_R {0.5 / EPS_R};
      real_t DIVIDE_EPS_T {0.5 / EPS_T};

      coord_t<D> xp_drp {xp[0] + EPS_R, xp[1]};
      coord_t<D> xp_drm {xp[0] - EPS_R, xp[1]};
      coord_t<D> xp_dtp {xp[0], xp[1] + EPS_T};
      coord_t<D> xp_dtm {xp[0], xp[1] - EPS_T};

      // iterate
      for (int i = 0; i < 10; i++) {

        // find midpoint values
        xpm[0] = HALF * (xp[0] + xpu[0]);
        xpm[1] = HALF * (xp[1] + xpu[1]);
        vm[0]  = HALF * (v[0] + vu[0]);
        vm[1]  = HALF * (v[1] + vu[1]);
        vm[2]  = HALF * (v[2] + vu[2]);

        // find contravariant midpoint velocity
        m_mblock.metric.v_Cov2Cntrv(xpm, vm, vmu);

        // find spacial derivatives
        real_t alphadr {DIVIDE_EPS_R
                        * (m_mblock.metric.alpha(xp_drp) - m_mblock.metric.alpha(xp_drm))};
        real_t betadr {DIVIDE_EPS_R
                       * (m_mblock.metric.beta1u(xp_drp) - m_mblock.metric.beta1u(xp_drm))};

        real_t g11dr {DIVIDE_EPS_R
                      * (m_mblock.metric.h_11_inv(xp_drp) - m_mblock.metric.h_11_inv(xp_drm))};
        real_t g22dr {DIVIDE_EPS_R
                      * (m_mblock.metric.h_22_inv(xp_drp) - m_mblock.metric.h_22_inv(xp_drm))};
        real_t g33dr {DIVIDE_EPS_R
                      * (m_mblock.metric.h_33_inv(xp_drp) - m_mblock.metric.h_33_inv(xp_drm))};

        real_t g33dt {DIVIDE_EPS_T
                      * (m_mblock.metric.h_33_inv(xp_dtp) - m_mblock.metric.h_33_inv(xp_dtm))};

        // find midpoint coefficients
        real_t gamma {math::sqrt(vm[0] * vmu[0] + vm[1] * vmu[1] + vm[2] * vmu[2])};
        real_t u0 {gamma / m_mblock.metric.alpha(xpm)};

        // find updated coordinate shift
        xpu[0] = xp[0] + m_dt * (vmu[0] / u0 - m_mblock.metric.beta1u(xpm));
        xpu[1] = xp[1] + m_dt * (vmu[1] / u0);

        // find updated velocity
        vu[0] = v[0]
                + m_dt
                    * (-m_mblock.metric.alpha(xpm) * u0 * alphadr + vm[0] * betadr
                       - 1 / (2 * u0)
                           * (g11dr * vm[0] * vm[0] + g22dr * vm[1] * vm[1]
                              + g33dr * vm[2] * vm[2]));

        vu[1] = v[1] + m_dt * (-1 / (2 * u0) * (g33dt * vm[2] * vm[2]));

        vu[2] = v[2];
      }

      // update coordinate
      int   I;
      float DX;
      Xi_TO_i_di(xpu[0], I, DX);
      m_particles.i1(p)  = I;
      m_particles.dx1(p) = DX;
      Xi_TO_i_di(xpu[1], I, DX);
      m_particles.i2(p)  = I;
      m_particles.dx2(p) = DX;

      // update velocity
      m_particles.ux1(p) = vu[0];
      m_particles.ux2(p) = vu[1];
      m_particles.ux3(p) = vu[2];
    }

    /**
     * Transform particle coordinate from code units i+di to `real_t` type.
     *
     * @param p index of the particle.
     * @param coord coordinate of the particle as a vector (of size D).
     */
    Inline void getParticleCoordinate(const index_t&, coord_t<D>&) const;

    /**
     * Update each velocity component.
     *
     * @param p index of the particle.
     * @param v corresponding 3-velocity component.
     */
    Inline void
    velocityUpdate(const index_t&, const real_t&, const real_t&, const real_t&) const;

    /**
     * Update each coordinate component.
     *
     * @param p index of the particle.
     * @param v corresponding 3-velocity component.
     */
    Inline void
    coordinateUpdate(const index_t&, const real_t&, const real_t&, const real_t&) const;

    /**
     * First order Yee mesh field interpolation to particle position.
     *
     * @param p index of the particle.
     * @param e interpolated e-field vector of size 3 [return].
     * @param b interpolated b-field vector of size 3 [return].
     */
    Inline void interpolateFields(const index_t&,
                                  vec_t<Dimension::THREE_D>&,
                                  vec_t<Dimension::THREE_D>&) const;

    /**
     * Boris algorithm.
     * @note Fields are modified inside the function and cannot be reused.
     *
     * @param p index of the particle.
     * @param e interpolated e-field vector of size 3 [modified].
     * @param b interpolated b-field vector of size 3 [modified].
     */
    Inline void
    BorisUpdate(const index_t&, vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
  };

  template <>
  Inline void
  Pusher<Dimension::ONE_D>::getParticleCoordinate(const index_t&             p,
                                                  coord_t<Dimension::ONE_D>& xp) const {
    xp[0] = static_cast<real_t>(m_particles.i1(p)) + static_cast<real_t>(m_particles.dx1(p));
  }
  template <>
  Inline void
  Pusher<Dimension::TWO_D>::getParticleCoordinate(const index_t&             p,
                                                  coord_t<Dimension::TWO_D>& xp) const {
    xp[0] = static_cast<real_t>(m_particles.i1(p)) + static_cast<real_t>(m_particles.dx1(p));
    xp[1] = static_cast<real_t>(m_particles.i2(p)) + static_cast<real_t>(m_particles.dx2(p));
  }
  template <>
  Inline void
  Pusher<Dimension::THREE_D>::getParticleCoordinate(const index_t&               p,
                                                    coord_t<Dimension::THREE_D>& xp) const {
    xp[0] = static_cast<real_t>(m_particles.i1(p)) + static_cast<real_t>(m_particles.dx1(p));
    xp[1] = static_cast<real_t>(m_particles.i2(p)) + static_cast<real_t>(m_particles.dx2(p));
    xp[2] = static_cast<real_t>(m_particles.i3(p)) + static_cast<real_t>(m_particles.dx3(p));
  }

  template <>
  Inline void Pusher<Dimension::TWO_D>::velocityUpdate(const index_t& p,
                                                       const real_t&  vx1,
                                                       const real_t&  vx2,
                                                       const real_t&  vx3) const {
    // get coordinate & velocity
    coord_t<Dimension::TWO_D> xp;
    getParticleCoordinate(p, xp);
    vec_t<Dimension::THREE_D> v {vx1, vx2, vx3};

    // initialize midpoint values & updated values
    coord_t<Dimension::TWO_D> xpm;
    coord_t<Dimension::TWO_D> xpu {xp[0], xp[1]};
    vec_t<Dimension::THREE_D> vm;
    vec_t<Dimension::THREE_D> vmu;
    vec_t<Dimension::THREE_D> vu {v[0], v[1], v[2]};

    // initialize coefficients & coordinates for derivatives
    real_t EPS_R {1e-6};
    real_t EPS_T {1e-6};
    real_t DIVIDE_EPS_R {0.5 / EPS_R};
    real_t DIVIDE_EPS_T {0.5 / EPS_T};

    coord_t<Dimension::TWO_D> xp_drp {xp[0] + EPS_R, xp[1]};
    coord_t<Dimension::TWO_D> xp_drm {xp[0] - EPS_R, xp[1]};
    coord_t<Dimension::TWO_D> xp_dtp {xp[0], xp[1] + EPS_T};
    coord_t<Dimension::TWO_D> xp_dtm {xp[0], xp[1] - EPS_T};

    // iterate
    for (int i = 0; i < 10; i++) {

      // find midpoint values
      xpm[0] = 0.5 * (xp[0] + xpu[0]);
      xpm[1] = 0.5 * (xp[1] + xpu[1]);
      vm[0]  = 0.5 * (v[0] + vu[0]);
      vm[1]  = 0.5 * (v[1] + vu[1]);
      vm[2]  = 0.5 * (v[2] + vu[2]);

      // find contravariant midpoint velocity
      m_mblock.metric.v_Cov2Cntrv(xpm, vm, vmu);

      // find spacial derivatives
      real_t alphadr {DIVIDE_EPS_R
                      * (m_mblock.metric.alpha(xp_drp) - m_mblock.metric.alpha(xp_drm))};
      real_t betadr {DIVIDE_EPS_R
                     * (m_mblock.metric.beta1u(xp_drp) - m_mblock.metric.beta1u(xp_drm))};

      real_t g11dr {DIVIDE_EPS_R
                    * (m_mblock.metric.h_11_inv(xp_drp) - m_mblock.metric.h_11_inv(xp_drm))};
      real_t g22dr {DIVIDE_EPS_R
                    * (m_mblock.metric.h_22_inv(xp_drp) - m_mblock.metric.h_22_inv(xp_drm))};
      real_t g33dr {DIVIDE_EPS_R
                    * (m_mblock.metric.h_33_inv(xp_drp) - m_mblock.metric.h_33_inv(xp_drm))};

      real_t g33dt {DIVIDE_EPS_T
                    * (m_mblock.metric.h_33_inv(xp_dtp) - m_mblock.metric.h_33_inv(xp_dtm))};

      // find midpoint coefficients
      real_t gamma {math::sqrt(1.0 + vm[0] * vmu[0] + vm[1] * vmu[1] + vm[2] * vmu[2])};
      real_t u0 {gamma / m_mblock.metric.alpha(xpm)};

      // find updated coordinate shift
      xpu[0] = xp[0] + m_dt * (vmu[0] / u0 - m_mblock.metric.beta1u(xpm));
      xpu[1] = xp[1] + m_dt * (vmu[1] / u0);

      // find updated velocity
      vu[0] = v[0]
              + m_dt
                  * (-m_mblock.metric.alpha(xpm) * u0 * alphadr + vm[0] * betadr
                     - 1 / (2 * u0)
                         * (g11dr * vm[0] * vm[0] + g22dr * vm[1] * vm[1]
                            + g33dr * vm[2] * vm[2]));

      vu[1] = v[1] + m_dt * (-1 / (2 * u0) * (g33dt * vm[2] * vm[2]));

      vu[2] = v[2];
    }

    // update velocity
    m_particles.ux1(p) = vu[0];
    m_particles.ux2(p) = vu[1];
    m_particles.ux3(p) = vu[2];
  }
  template <>
  Inline void Pusher<Dimension::THREE_D>::velocityUpdate(const index_t&,
                                                         const real_t&,
                                                         const real_t&,
                                                         const real_t&) const {}

  template <>
  Inline void Pusher<Dimension::TWO_D>::coordinateUpdate(const index_t& p,
                                                         const real_t&  vx1,
                                                         const real_t&  vx2,
                                                         const real_t&  vx3) const {
    // get coordinate & velocity
    coord_t<Dimension::TWO_D> xp;
    getParticleCoordinate(p, xp);
    vec_t<Dimension::THREE_D> v {vx1, vx2, vx3};
    vec_t<Dimension::THREE_D> vu {v[0], v[1], v[2]};
    real_t                    gamma {math::sqrt(v[0] * vu[0] + v[1] * vu[1] + v[2] * vu[2])};

    // initialize midpoint values & updated values
    coord_t<Dimension::TWO_D> xpm;
    coord_t<Dimension::TWO_D> xpu {xp[0], xp[1]};

    // iterate
    for (int i = 0; i < 10; i++) {

      // find midpoint values
      xpm[0] = 0.5 * (xp[0] + xpu[0]);
      xpm[1] = 0.5 * (xp[1] + xpu[1]);

      // find contravariant midpoint velocity
      m_mblock.metric.v_Cov2Cntrv(xpm, v, vu);

      // find midpoint coefficients
      real_t u0 {gamma / m_mblock.metric.alpha(xpm)};

      // find updated coordinate shift
      xpu[0] = xp[0] + m_dt * (vu[0] / u0 - m_mblock.metric.beta1u(xpm));
      xpu[1] = xp[1] + m_dt * (vu[1] / u0);
    }

    // update coordinate
    int   I;
    float DX;
    Xi_TO_i_di(xpu[0], I, DX);
    m_particles.i1(p)  = I;
    m_particles.dx1(p) = DX;
    Xi_TO_i_di(xpu[1], I, DX);
    m_particles.i2(p)  = I;
    m_particles.dx2(p) = DX;
  }

  template <>
  Inline void Pusher<Dimension::THREE_D>::coordinateUpdate(const index_t&,
                                                           const real_t&,
                                                           const real_t&,
                                                           const real_t&) const {}

  // * * * * * * * * * * * * * * *
  // Boris velocity update
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Pusher<Dimension::TWO_D>::BorisUpdate(const index_t&             p,
                                                    vec_t<Dimension::THREE_D>& d0,
                                                    vec_t<Dimension::THREE_D>& b0) const {

    coord_t<Dimension::TWO_D> xp;
    getParticleCoordinate(p, xp);
    vec_t<Dimension::THREE_D> vv;
    vec_t<Dimension::THREE_D> vu;
    m_mblock.metric.v_Cov2Hat(
      xp, {m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p)}, vv);
    vv[0] += d0[0];
    vv[1] += d0[1];
    vv[2] += d0[2];
    const real_t gamma {ONE / math::sqrt(ONE + vv[0] * vv[0] + vv[1] * vv[1] + vv[2] * vv[2])};
    const real_t prefactor {m_dt / TWO * m_mblock.metric.alpha(xp)};
    vec_t<Dimension::THREE_D> tt {
      prefactor * gamma * b0[0], prefactor * gamma * b0[1], prefactor * gamma * b0[2]};
    const real_t ff {ONE / math::sqrt(ONE + tt[0] * tt[0] + tt[1] * tt[1] + tt[2] * tt[2])};

    vu[0] = ff * (vv[0] + vv[1] * tt[2] - vv[2] * tt[1]);
    vu[1] = ff * (vv[1] + vv[2] * tt[0] - vv[0] * tt[2]);
    vu[2] = ff * (vv[2] + vv[0] * tt[1] - vv[1] * tt[0]);

    m_mblock.metric.v_Hat2Cov(xp, vu, vv);
    m_particles.ux1(p) = vv[0];
    m_particles.ux2(p) = vv[1];
    m_particles.ux3(p) = vv[2];
  }
  template <>
  Inline void Pusher<Dimension::THREE_D>::BorisUpdate(const index_t&,
                                                      vec_t<Dimension::THREE_D>&,
                                                      vec_t<Dimension::THREE_D>&) const {}

  // * * * * * * * * * * * * * * *
  // Field interpolations
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Pusher<Dimension::TWO_D>::interpolateFields(
    const index_t& p, vec_t<Dimension::THREE_D>& e0, vec_t<Dimension::THREE_D>& b0) const {
    const auto   i {m_particles.i1(p) + N_GHOSTS};
    const real_t dx1 {static_cast<real_t>(m_particles.dx1(p))};
    const auto   j {m_particles.i2(p) + N_GHOSTS};
    const real_t dx2 {static_cast<real_t>(m_particles.dx2(p))};

    // first order
    real_t c000, c100, c010, c110, c00, c10;

    // Dx1
    // interpolate to nodes
    c000 = HALF * (m_mblock.em(i, j, em::ex1) + m_mblock.em(i - 1, j, em::ex1));
    c100 = HALF * (m_mblock.em(i, j, em::ex1) + m_mblock.em(i + 1, j, em::ex1));
    c010 = HALF * (m_mblock.em(i, j + 1, em::ex1) + m_mblock.em(i - 1, j + 1, em::ex1));
    c110 = HALF * (m_mblock.em(i, j + 1, em::ex1) + m_mblock.em(i + 1, j + 1, em::ex1));
    // interpolate from nodes to the particle position
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    e0[0] = c00 * (ONE - dx2) + c10 * dx2;
    // Dx2
    c000  = HALF * (m_mblock.em(i, j, em::ex2) + m_mblock.em(i, j - 1, em::ex2));
    c100  = HALF * (m_mblock.em(i + 1, j, em::ex2) + m_mblock.em(i + 1, j - 1, em::ex2));
    c010  = HALF * (m_mblock.em(i, j, em::ex2) + m_mblock.em(i, j + 1, em::ex2));
    c110  = HALF * (m_mblock.em(i + 1, j, em::ex2) + m_mblock.em(i + 1, j + 1, em::ex2));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    e0[1] = c00 * (ONE - dx2) + c10 * dx2;
    // Dx3
    c000  = m_mblock.em(i, j, em::ex3);
    c100  = m_mblock.em(i + 1, j, em::ex3);
    c010  = m_mblock.em(i, j + 1, em::ex3);
    c110  = m_mblock.em(i + 1, j + 1, em::ex3);
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    e0[2] = c00 * (ONE - dx2) + c10 * dx2;

    // Bx1
    c000  = HALF * (m_mblock.em0(i, j, em::bx1) + m_mblock.em0(i, j - 1, em::bx1));
    c100  = HALF * (m_mblock.em0(i + 1, j, em::bx1) + m_mblock.em0(i + 1, j - 1, em::bx1));
    c010  = HALF * (m_mblock.em0(i, j, em::bx1) + m_mblock.em0(i, j + 1, em::bx1));
    c110  = HALF * (m_mblock.em0(i + 1, j, em::bx1) + m_mblock.em0(i + 1, j + 1, em::bx1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    b0[0] = c00 * (ONE - dx2) + c10 * dx2;
    // Bx2
    c000  = HALF * (m_mblock.em0(i - 1, j, em::bx2) + m_mblock.em0(i, j, em::bx2));
    c100  = HALF * (m_mblock.em0(i, j, em::bx2) + m_mblock.em0(i + 1, j, em::bx2));
    c010  = HALF * (m_mblock.em0(i - 1, j + 1, em::bx2) + m_mblock.em0(i, j + 1, em::bx2));
    c110  = HALF * (m_mblock.em0(i, j + 1, em::bx2) + m_mblock.em0(i + 1, j + 1, em::bx2));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    b0[1] = c00 * (ONE - dx2) + c10 * dx2;
    // Bx3
    c000 = QUARTER
           * (m_mblock.em0(i - 1, j - 1, em::bx3) + m_mblock.em0(i - 1, j, em::bx3)
              + m_mblock.em0(i, j - 1, em::bx3) + m_mblock.em0(i, j, em::bx3));
    c100 = QUARTER
           * (m_mblock.em0(i, j - 1, em::bx3) + m_mblock.em0(i, j, em::bx3)
              + m_mblock.em0(i + 1, j - 1, em::bx3) + m_mblock.em0(i + 1, j, em::bx3));
    c010 = QUARTER
           * (m_mblock.em0(i - 1, j, em::bx3) + m_mblock.em0(i - 1, j + 1, em::bx3)
              + m_mblock.em0(i, j, em::bx3) + m_mblock.em0(i, j + 1, em::bx3));
    c110 = QUARTER
           * (m_mblock.em0(i, j, em::bx3) + m_mblock.em0(i, j + 1, em::bx3)
              + m_mblock.em0(i + 1, j, em::bx3) + m_mblock.em0(i + 1, j + 1, em::bx3));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    b0[2] = c00 * (ONE - dx2) + c10 * dx2;
  }
  template <>
  Inline void Pusher<Dimension::THREE_D>::interpolateFields(const index_t&,
                                                            vec_t<Dimension::THREE_D>&,
                                                            vec_t<Dimension::THREE_D>&) const {
  }

} // namespace ntt

#endif
