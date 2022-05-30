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
    real_t                            m_coeff, m_dt;

  public:
    Pusher(const Meshblock<D, SimulationType::GRPIC>& mblock,
           const Particles<D, SimulationType::GRPIC>& particles,
           const real_t&                            coeff,
           const real_t&                            dt)
      : m_mblock(mblock), m_particles(particles), m_coeff(coeff), m_dt(dt) {}
    /**
     * Loop over all active particles of the given species and call the appropriate pusher.
     * TODO: forward/backward
     */
    void pushParticles() {
      if (m_particles.pusher() == ParticlePusher::PHOTON) {
        // push photons
        auto range_policy = Kokkos::RangePolicy<AccelExeSpace, Photon_t>(0, m_particles.npart());
        Kokkos::parallel_for("pusher", range_policy, *this);
      }
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
      coord_t<D> xpm;
      coord_t<D> xpu {xp[0], xp[1]};
      vec_t<Dimension::THREE_D> vm;
      vec_t<Dimension::THREE_D> vmu;
      vec_t<Dimension::THREE_D> vu {v[0], v[1], v[2]};
      
      // initialize coefficients & coordinates for derivatives
      real_t EPS_R {1e-6};
      real_t EPS_T {1e-6};
      real_t DIVIDE_EPS_R {0.5 / EPS_R};
      real_t DIVIDE_EPS_T {0.5 / EPS_T};

      coord_t<D> xp_drp {xp[0]+EPS_R, xp[1]};
      coord_t<D> xp_drm {xp[0]-EPS_R, xp[1]};
      coord_t<D> xp_dtp {xp[0], xp[1]+EPS_T};
      coord_t<D> xp_dtm {xp[0], xp[1]-EPS_T};

      // iterate
      for (int i = 0; i < 10; i++) {

        // find midpoint values
        xpm[0] = 0.5*(xp[0]+xpu[0]);
        xpm[1] = 0.5*(xp[1]+xpu[1]);
        vm[0] = 0.5*(v[0]+vu[0]);
        vm[1] = 0.5*(v[1]+vu[1]);
        vm[2] = 0.5*(v[2]+vu[2]);

        // find contravariant midpoint velocity
        m_mblock.metric.v_Cov2Cntrv(xpm, vm, vmu);

        // find spacial derivatives
        real_t alphadr {DIVIDE_EPS_R * (m_mblock.metric.alpha(xp_drp) - m_mblock.metric.alpha(xp_drm))};
        real_t betadr {DIVIDE_EPS_R * (m_mblock.metric.beta1u(xp_drp) - m_mblock.metric.beta1u(xp_drm))};

        real_t g11dr {DIVIDE_EPS_R * (m_mblock.metric.h_11_inv(xp_drp) - m_mblock.metric.h_11_inv(xp_drm))};
        real_t g22dr {DIVIDE_EPS_R * (m_mblock.metric.h_22_inv(xp_drp) - m_mblock.metric.h_22_inv(xp_drm))};
        real_t g33dr {DIVIDE_EPS_R * (m_mblock.metric.h_33_inv(xp_drp) - m_mblock.metric.h_33_inv(xp_drm))};

        real_t g33dt {DIVIDE_EPS_T * (m_mblock.metric.h_33_inv(xp_dtp) - m_mblock.metric.h_33_inv(xp_dtm))};

        // find midpoint coefficients
        real_t gamma {math::sqrt(vm[0]*vmu[0] + vm[1]*vmu[1] + vm[2]*vmu[2])};
        real_t u0 {gamma/m_mblock.metric.alpha(xpm)};

        // find updated coordinate shift
        xpu[0] = xp[0] + m_dt * (m_mblock.metric.alpha(xpm) * vmu[0] / u0 - m_mblock.metric.beta1u(xpm));
        xpu[1] = xp[1] + m_dt * (m_mblock.metric.alpha(xpm) * vmu[1] / u0);

        // find updated velocity
        vu[0] = v[0] + m_dt * ( -m_mblock.metric.alpha(xpm)*u0*alphadr + vm[0]*betadr
            - 1/(2*u0)*(g11dr*vm[0]*vm[0] + g22dr*vm[1]*vm[1] + g33dr*vm[2]*vm[2]));

        vu[1] = v[1] + m_dt * ( 1/(2*u0)*(g33dt*vm[2]*vm[2]) );

        vu[2] = v[2];
      }
      
      // update coordinate
      m_particles.dx1(p) = m_particles.dx1(p) + static_cast<float>(xpu[0] - xp[0]);
      int   temp_i {static_cast<int>(m_particles.dx1(p))};
      float temp_r {math::fmax(SIGNf(m_particles.dx1(p)) + temp_i, static_cast<float>(temp_i)) - 1.0f};
      temp_i             = static_cast<int>(temp_r);
      m_particles.i1(p)  = m_particles.i1(p) + temp_i;
      m_particles.dx1(p) = m_particles.dx1(p) - temp_r;

      m_particles.dx2(p) = m_particles.dx2(p) + static_cast<float>(xpu[1] - xp[1]);
      temp_i = static_cast<int>(m_particles.dx2(p));
      temp_r = math::fmax(SIGNf(m_particles.dx2(p)) + temp_i, static_cast<float>(temp_i)) - 1.0f;
      temp_i             = static_cast<int>(temp_r);
      m_particles.i2(p)  = m_particles.i2(p) + temp_i;
      m_particles.dx2(p) = m_particles.dx2(p) - temp_r;
      
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
     * Update particle velocities.
     *
     * @param p index of the particle.
     * @param v particle current 3-velocity.
     */
    Inline void velocityUpdate(const index_t&, const vec_t<Dimension::THREE_D>&) const;


    /**
     * Update particle positions according to updated velocities.
     *
     * @param p index of the particle.
     * @param v particle 3-velocity.
     */
    Inline void positionUpdate(const index_t&, const vec_t<Dimension::THREE_D>&) const;

    /**
     * Update each position component.
     *
     * @param p index of the particle.
     * @param v corresponding 3-velocity component.
     */
    Inline void positionUpdate_x1(const index_t&, const real_t&) const;
    Inline void positionUpdate_x2(const index_t&, const real_t&) const;
    Inline void positionUpdate_x3(const index_t&, const real_t&) const;

    /**
     * Update each velocity component.
     *
     * @param p index of the particle.
     * @param v corresponding 3-velocity component.
     */
    Inline void velocityUpdate_v123(const index_t&, const real_t&, const real_t&, const real_t&) const;

  };

  // * * * * * * * * * * * * * * *
  // General velocity update
  // * * * * * * * * * * * * * * *
  //template <>
  //Inline void Pusher<Dimension::ONE_D>::velocityUpdate(const index_t& p, const vec_t<Dimension::THREE_D>& v) const {
  //  velocityUpdate_v123(p, v[0], v[1], v[2]);
  //}
  template <>
  Inline void Pusher<Dimension::TWO_D>::velocityUpdate(const index_t& p, const vec_t<Dimension::THREE_D>& v) const {
    velocityUpdate_v123(p, v[0], v[1], v[2]);
  }
  //template <>
  //Inline void Pusher<Dimension::THREE_D>::velocityUpdate(const index_t& p, const vec_t<Dimension::THREE_D>& v) const {
  //  velocityUpdate_v123(p, v[0], v[1], v[2]);
  //}

  // * * * * * * * * * * * * * * *
  // General position update
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Pusher<Dimension::ONE_D>::positionUpdate(const index_t& p, const vec_t<Dimension::THREE_D>& v) const {
    positionUpdate_x1(p, v[0]);
  }
  template <>
  Inline void Pusher<Dimension::TWO_D>::positionUpdate(const index_t& p, const vec_t<Dimension::THREE_D>& v) const {
    positionUpdate_x1(p, v[0]);
    positionUpdate_x2(p, v[1]);
  }
  template <>
  Inline void Pusher<Dimension::THREE_D>::positionUpdate(const index_t& p, const vec_t<Dimension::THREE_D>& v) const {
    positionUpdate_x1(p, v[0]);
    positionUpdate_x2(p, v[1]);
    positionUpdate_x3(p, v[2]);
  }

  template <>
  Inline void Pusher<Dimension::ONE_D>::getParticleCoordinate(const index_t& p, coord_t<Dimension::ONE_D>& xp) const {
    xp[0] = static_cast<real_t>(m_particles.i1(p)) + static_cast<real_t>(m_particles.dx1(p));
  }
  template <>
  Inline void Pusher<Dimension::TWO_D>::getParticleCoordinate(const index_t& p, coord_t<Dimension::TWO_D>& xp) const {
    xp[0] = static_cast<real_t>(m_particles.i1(p)) + static_cast<real_t>(m_particles.dx1(p));
    xp[1] = static_cast<real_t>(m_particles.i2(p)) + static_cast<real_t>(m_particles.dx2(p));
  }
  template <>
  Inline void Pusher<Dimension::THREE_D>::getParticleCoordinate(const index_t&               p,
                                                                coord_t<Dimension::THREE_D>& xp) const {
    xp[0] = static_cast<real_t>(m_particles.i1(p)) + static_cast<real_t>(m_particles.dx1(p));
    xp[1] = static_cast<real_t>(m_particles.i2(p)) + static_cast<real_t>(m_particles.dx2(p));
    xp[2] = static_cast<real_t>(m_particles.i3(p)) + static_cast<real_t>(m_particles.dx3(p));
  }

  template <Dimension D>
  Inline void Pusher<D>::positionUpdate_x1(const index_t& p, const real_t& vx1) const {
    m_particles.dx1(p) = m_particles.dx1(p) + static_cast<float>(m_dt * vx1);
    int   temp_i {static_cast<int>(m_particles.dx1(p))};
    float temp_r {math::fmax(SIGNf(m_particles.dx1(p)) + temp_i, static_cast<float>(temp_i)) - 1.0f};
    temp_i             = static_cast<int>(temp_r);
    m_particles.i1(p)  = m_particles.i1(p) + temp_i;
    m_particles.dx1(p) = m_particles.dx1(p) - temp_r;
  }
  template <Dimension D>
  Inline void Pusher<D>::positionUpdate_x2(const index_t& p, const real_t& vx2) const {
    m_particles.dx2(p) = m_particles.dx2(p) + static_cast<float>(m_dt * vx2);
    int   temp_i {static_cast<int>(m_particles.dx2(p))};
    float temp_r {math::fmax(SIGNf(m_particles.dx2(p)) + temp_i, static_cast<float>(temp_i)) - 1.0f};
    temp_i             = static_cast<int>(temp_r);
    m_particles.i2(p)  = m_particles.i2(p) + temp_i;
    m_particles.dx2(p) = m_particles.dx2(p) - temp_r;
  }
  template <Dimension D>
  Inline void Pusher<D>::positionUpdate_x3(const index_t& p, const real_t& vx3) const {
    m_particles.dx3(p) = m_particles.dx3(p) + static_cast<float>(m_dt * vx3);
    int   temp_i {static_cast<int>(m_particles.dx3(p))};
    float temp_r {math::fmax(SIGNf(m_particles.dx3(p)) + temp_i, static_cast<float>(temp_i)) - 1.0f};
    temp_i             = static_cast<int>(temp_r);
    m_particles.i3(p)  = m_particles.i3(p) + temp_i;
    m_particles.dx3(p) = m_particles.dx3(p) - temp_r;
  }

  template <Dimension D>
  Inline void Pusher<D>::velocityUpdate_v123(const index_t& p, const real_t& vx1, const real_t& vx2, const real_t& vx3) const {
    // get coordinate & velocity
    coord_t<D> xp;
    getParticleCoordinate(p, xp);
    vec_t<Dimension::THREE_D> v {vx1, vx2, vx3};
      
    // initialize midpoint values & updated values
    coord_t<D> xpm;
    coord_t<D> xpu {xp[0], xp[1]};
    vec_t<Dimension::THREE_D> vm;
    vec_t<Dimension::THREE_D> vmu;
    vec_t<Dimension::THREE_D> vu {v[0], v[1], v[2]};
      
    // initialize coefficients & coordinates for derivatives
    real_t EPS_R {1e-6};
    real_t EPS_T {1e-6};
    real_t DIVIDE_EPS_R {0.5 / EPS_R};
    real_t DIVIDE_EPS_T {0.5 / EPS_T};

    coord_t<D> xp_drp {xp[0]+EPS_R, xp[1]};
    coord_t<D> xp_drm {xp[0]-EPS_R, xp[1]};
    coord_t<D> xp_dtp {xp[0], xp[1]+EPS_T};
    coord_t<D> xp_dtm {xp[0], xp[1]-EPS_T};

    // iterate
    for (int i = 0; i < 10; i++) {

      // find midpoint values
      xpm[0] = 0.5*(xp[0]+xpu[0]);
      xpm[1] = 0.5*(xp[1]+xpu[1]);
      vm[0] = 0.5*(v[0]+vu[0]);
      vm[1] = 0.5*(v[1]+vu[1]);
      vm[2] = 0.5*(v[2]+vu[2]);

      // find contravariant midpoint velocity
      m_mblock.metric.v_Cov2Cntrv(xpm, vm, vmu);

      // find spacial derivatives
      real_t alphadr {DIVIDE_EPS_R * (m_mblock.metric.alpha(xp_drp) - m_mblock.metric.alpha(xp_drm))};
      real_t betadr {DIVIDE_EPS_R * (m_mblock.metric.beta1u(xp_drp) - m_mblock.metric.beta1u(xp_drm))};

      real_t g11dr {DIVIDE_EPS_R * (m_mblock.metric.h_11(xp_drp) - m_mblock.metric.h_11(xp_drm))};
      real_t g22dr {DIVIDE_EPS_R * (m_mblock.metric.h_22(xp_drp) - m_mblock.metric.h_22(xp_drm))};
      real_t g33dr {DIVIDE_EPS_R * (m_mblock.metric.h_33(xp_drp) - m_mblock.metric.h_33(xp_drm))};

      real_t g33dt {DIVIDE_EPS_T * (m_mblock.metric.h_33(xp_dtp) - m_mblock.metric.h_33(xp_dtm))};

      // find midpoint coefficients
      real_t gamma {math::sqrt(1 + vm[0]*vmu[0] + vm[1]*vmu[1] + vm[2]*vmu[2])};
      real_t u0 {gamma/m_mblock.metric.alpha(xpm)};

      // find updated velocity
      vu[0] = v[0] + m_dt * ( -m_mblock.metric.alpha(xpm)*u0*alphadr + vm[0]*betadr
          - 1/(2*u0)*(g11dr*vm[0]*vm[0] + g22dr*vm[1]*vm[1] + g33dr*vm[2]*vm[2])  );

      vu[1] = v[1] + m_dt * ( 1/(2*u0)*(g33dt*vm[2]*vm[2]) );

      vu[2] = v[2];
    }
    // update velocity
    m_particles.ux1(p) = vu[0];
    m_particles.ux2(p) = vu[1];
    m_particles.ux3(p) = vu[2];
  }

} // namespace ntt

#endif
