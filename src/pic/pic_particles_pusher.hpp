#ifndef PIC_PARTICLES_PUSHER_H
#define PIC_PARTICLES_PUSHER_H

#include "global.h"
#include "fields.h"
#include "particles.h"
#include "meshblock.h"
#include "pic.h"

#include <stdexcept>

namespace ntt {
  struct BorisFwd_t {};
  struct BorisBwd_t {};
  struct Photon_t {};

  /**
   * @brief Algorithm for the Particle pusher.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Pusher {
    Meshblock<D, SimulationType::PIC> m_mblock;
    Particles<D, SimulationType::PIC> m_particles;
    real_t                            m_coeff, m_dt;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param particles Particles.
     * @param coeff Coefficient to be multiplied by dE/dt = coeff * curl B.
     * @param dt Time step.
     */
    Pusher(const Meshblock<D, SimulationType::PIC>& mblock,
           const Particles<D, SimulationType::PIC>& particles,
           const real_t&                            coeff,
           const real_t&                            dt)
      : m_mblock(mblock), m_particles(particles), m_coeff(coeff), m_dt(dt) {}
    /**
     * @brief Loop over all active particles of the given species and call the appropriate pusher.
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
            = Kokkos::RangePolicy<AccelExeSpace, BorisFwd_t>(0, m_particles.npart());
          Kokkos::parallel_for("pusher", range_policy, *this);
        } else {
          //// push backward
          // auto range_policy
          //   = Kokkos::RangePolicy<AccelExeSpace, BorisBwd_t>(0, m_particles.npart());
          // Kokkos::parallel_for("pusher", range_policy, *this);
        }
      } else {
        NTTError("pusher not implemented");
      }
    }
    /**
     * @brief Pusher for the forward Boris algorithm.
     * @param p index.
     */
    Inline void operator()(const BorisFwd_t&, index_t p) const {
      vec_t<Dimension::THREE_D> e_int, b_int, e_int_Cart, b_int_Cart;
      interpolateFields(p, e_int, b_int);

      coord_t<D> xp;
      getParticleCoordinate(p, xp);
      m_mblock.metric.v_Cntrv2Cart(xp, e_int, e_int_Cart);
      m_mblock.metric.v_Cntrv2Cart(xp, b_int, b_int_Cart);

      BorisUpdate(p, e_int_Cart, b_int_Cart);

      real_t inv_energy;
      inv_energy = SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p)) + SQR(m_particles.ux3(p));
      inv_energy = ONE / math::sqrt(ONE + inv_energy);

      vec_t<Dimension::THREE_D> v;
      m_mblock.metric.v_Cart2Cntrv(
        xp, {m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p)}, v);
      v[0] *= inv_energy;
      v[1] *= inv_energy;
      v[2] *= inv_energy;
      positionUpdate(p, v);
    }
    /**
     * @brief Pusher for the photon pusher.
     * @param p index.
     */
    Inline void operator()(const Photon_t&, index_t p) const {
      coord_t<D> xp;
      getParticleCoordinate(p, xp);
      vec_t<Dimension::THREE_D> v;
      m_mblock.metric.v_Cart2Cntrv(
        xp, {m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p)}, v);
      real_t inv_energy;
      inv_energy = SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p)) + SQR(m_particles.ux3(p));
      inv_energy = ONE / math::sqrt(inv_energy);
      v[0] *= inv_energy;
      v[1] *= inv_energy;
      v[2] *= inv_energy;
      positionUpdate(p, v);
    }
    // Inline void operator()(const BorisBwd_t&, index_t p) const {
    //   real_t inv_energy;
    //   inv_energy = SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p)) +
    //   SQR(m_particles.ux3(p)); inv_energy = ONE / math::sqrt(ONE + inv_energy);

    //   coord_t<D> xp;
    //   getParticleCoordinate(p, xp);

    //   vec_t<Dimension::THREE_D> v;
    //   m_mblock.metric.v_Cart2Cntrv(
    //     xp, {m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p)}, v);
    //   v[0] *= inv_energy;
    //   v[1] *= inv_energy;
    //   v[2] *= inv_energy;
    //   positionUpdate(p, v);
    //   getParticleCoordinate(p, xp);

    //   vec_t<Dimension::THREE_D> e_int, b_int, e_int_Cart, b_int_Cart;
    //   interpolateFields(p, e_int, b_int);

    //   m_mblock.metric.v_Cntrv2Cart(xp, e_int, e_int_Cart);
    //   m_mblock.metric.v_Cntrv2Cart(xp, b_int, b_int_Cart);

    //   BorisUpdate(p, e_int_Cart, b_int_Cart);
    // }

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
    Inline void interpolateFields(index_t&,
                                  vec_t<Dimension::THREE_D>&,
                                  vec_t<Dimension::THREE_D>&) const;

    /**
     * @brief Update particle positions according to updated velocities.
     * @param p index of the particle.
     * @param v particle 3-velocity.
     */
    Inline void positionUpdate(index_t&, const vec_t<Dimension::THREE_D>&) const;

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
    Inline void
    BorisUpdate(index_t&, vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
  };

  template <>
  Inline void
  Pusher<Dimension::ONE_D>::getParticleCoordinate(index_t&             p,
                                                  coord_t<Dimension::ONE_D>& xp) const {
    xp[0] = static_cast<real_t>(m_particles.i1(p)) + static_cast<real_t>(m_particles.dx1(p));
  }
  template <>
  Inline void
  Pusher<Dimension::TWO_D>::getParticleCoordinate(index_t&             p,
                                                  coord_t<Dimension::TWO_D>& xp) const {
    xp[0] = static_cast<real_t>(m_particles.i1(p)) + static_cast<real_t>(m_particles.dx1(p));
    xp[1] = static_cast<real_t>(m_particles.i2(p)) + static_cast<real_t>(m_particles.dx2(p));
  }
  template <>
  Inline void
  Pusher<Dimension::THREE_D>::getParticleCoordinate(index_t&               p,
                                                    coord_t<Dimension::THREE_D>& xp) const {
    xp[0] = static_cast<real_t>(m_particles.i1(p)) + static_cast<real_t>(m_particles.dx1(p));
    xp[1] = static_cast<real_t>(m_particles.i2(p)) + static_cast<real_t>(m_particles.dx2(p));
    xp[2] = static_cast<real_t>(m_particles.i3(p)) + static_cast<real_t>(m_particles.dx3(p));
  }

  // * * * * * * * * * * * * * * *
  // General position update
  // * * * * * * * * * * * * * * *
  template <>
  Inline void
  Pusher<Dimension::ONE_D>::positionUpdate(index_t&                   p,
                                           const vec_t<Dimension::THREE_D>& v) const {
    positionUpdate_x1(p, v[0]);
  }
  template <>
  Inline void
  Pusher<Dimension::TWO_D>::positionUpdate(index_t&                   p,
                                           const vec_t<Dimension::THREE_D>& v) const {
    positionUpdate_x1(p, v[0]);
    positionUpdate_x2(p, v[1]);
  }
  template <>
  Inline void
  Pusher<Dimension::THREE_D>::positionUpdate(index_t&                   p,
                                             const vec_t<Dimension::THREE_D>& v) const {
    positionUpdate_x1(p, v[0]);
    positionUpdate_x2(p, v[1]);
    positionUpdate_x3(p, v[2]);
  }

  template <Dimension D>
  Inline void Pusher<D>::positionUpdate_x1(index_t& p, const real_t& vx1) const {
    m_particles.dx1(p) = m_particles.dx1(p) + static_cast<float>(m_dt * vx1);
    int   temp_i {static_cast<int>(m_particles.dx1(p))};
    float temp_r {math::fmax(SIGNf(m_particles.dx1(p)) + temp_i, static_cast<float>(temp_i))
                  - 1.0f};
    temp_i             = static_cast<int>(temp_r);
    m_particles.i1(p)  = m_particles.i1(p) + temp_i;
    m_particles.dx1(p) = m_particles.dx1(p) - temp_r;
  }
  template <Dimension D>
  Inline void Pusher<D>::positionUpdate_x2(index_t& p, const real_t& vx2) const {
    m_particles.dx2(p) = m_particles.dx2(p) + static_cast<float>(m_dt * vx2);
    int   temp_i {static_cast<int>(m_particles.dx2(p))};
    float temp_r {math::fmax(SIGNf(m_particles.dx2(p)) + temp_i, static_cast<float>(temp_i))
                  - 1.0f};
    temp_i             = static_cast<int>(temp_r);
    m_particles.i2(p)  = m_particles.i2(p) + temp_i;
    m_particles.dx2(p) = m_particles.dx2(p) - temp_r;
  }
  template <Dimension D>
  Inline void Pusher<D>::positionUpdate_x3(index_t& p, const real_t& vx3) const {
    m_particles.dx3(p) = m_particles.dx3(p) + static_cast<float>(m_dt * vx3);
    int   temp_i {static_cast<int>(m_particles.dx3(p))};
    float temp_r {math::fmax(SIGNf(m_particles.dx3(p)) + temp_i, static_cast<float>(temp_i))
                  - 1.0f};
    temp_i             = static_cast<int>(temp_r);
    m_particles.i3(p)  = m_particles.i3(p) + temp_i;
    m_particles.dx3(p) = m_particles.dx3(p) - temp_r;
  }

  // * * * * * * * * * * * * * * *
  // Boris velocity update
  // * * * * * * * * * * * * * * *
  template <Dimension D>
  Inline void Pusher<D>::BorisUpdate(index_t&             p,
                                     vec_t<Dimension::THREE_D>& e0,
                                     vec_t<Dimension::THREE_D>& b0) const {
    real_t COEFF {m_coeff};

    e0[0] *= COEFF;
    e0[1] *= COEFF;
    e0[2] *= COEFF;
    vec_t<Dimension::THREE_D> u0 {
      m_particles.ux1(p) + e0[0], m_particles.ux2(p) + e0[1], m_particles.ux3(p) + e0[2]};

    COEFF *= 1.0 / math::sqrt(1.0 + u0[0] * u0[0] + u0[1] * u0[1] + u0[2] * u0[2]);
    b0[0] *= COEFF;
    b0[1] *= COEFF;
    b0[2] *= COEFF;
    COEFF = 2.0 / (1.0 + b0[0] * b0[0] + b0[1] * b0[1] + b0[2] * b0[2]);

    vec_t<Dimension::THREE_D> u1 {(u0[0] + u0[1] * b0[2] - u0[2] * b0[1]) * COEFF,
                                  (u0[1] + u0[2] * b0[0] - u0[0] * b0[2]) * COEFF,
                                  (u0[2] + u0[0] * b0[1] - u0[1] * b0[0]) * COEFF};

    u0[0] += u1[1] * b0[2] - u1[2] * b0[1] + e0[0];
    u0[1] += u1[2] * b0[0] - u1[0] * b0[2] + e0[1];
    u0[2] += u1[0] * b0[1] - u1[1] * b0[0] + e0[2];

    m_particles.ux1(p) = u0[0];
    m_particles.ux2(p) = u0[1];
    m_particles.ux3(p) = u0[2];
  }

  // * * * * * * * * * * * * * * *
  // Field interpolations
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Pusher<Dimension::ONE_D>::interpolateFields(
    index_t& p, vec_t<Dimension::THREE_D>& e0, vec_t<Dimension::THREE_D>& b0) const {
    const auto   i {m_particles.i1(p) + N_GHOSTS};
    const real_t dx1 {static_cast<real_t>(m_particles.dx1(p))};

    // first order
    real_t c0, c1;

    // Ex1
    // interpolate to nodes
    c0 = HALF * (m_mblock.em(i, em::ex1) + m_mblock.em(i - 1, em::ex1));
    c1 = HALF * (m_mblock.em(i, em::ex1) + m_mblock.em(i + 1, em::ex1));
    // interpolate from nodes to the particle position
    e0[0] = c0 * (ONE - dx1) + c1 * dx1;
    // Ex2
    c0    = m_mblock.em(i, em::ex2);
    c1    = m_mblock.em(i + 1, em::ex2);
    e0[1] = c0 * (ONE - dx1) + c1 * dx1;
    // Ex3
    c0    = m_mblock.em(i, em::ex3);
    c1    = m_mblock.em(i + 1, em::ex3);
    e0[2] = c0 * (ONE - dx1) + c1 * dx1;

    // Bx1
    c0    = m_mblock.em(i, em::bx1);
    c1    = m_mblock.em(i + 1, em::bx1);
    b0[0] = c0 * (ONE - dx1) + c1 * dx1;
    // Bx2
    c0    = HALF * (m_mblock.em(i - 1, em::bx2) + m_mblock.em(i, em::bx2));
    c1    = HALF * (m_mblock.em(i, em::bx2) + m_mblock.em(i + 1, em::bx2));
    b0[1] = c0 * (ONE - dx1) + c1 * dx1;
    // Bx3
    c0    = HALF * (m_mblock.em(i - 1, em::bx3) + m_mblock.em(i, em::bx3));
    c1    = HALF * (m_mblock.em(i, em::bx3) + m_mblock.em(i + 1, em::bx3));
    b0[2] = c0 * (ONE - dx1) + c1 * dx1;
  }

  template <>
  Inline void Pusher<Dimension::TWO_D>::interpolateFields(
    index_t& p, vec_t<Dimension::THREE_D>& e0, vec_t<Dimension::THREE_D>& b0) const {
    const auto   i {m_particles.i1(p) + N_GHOSTS};
    const real_t dx1 {static_cast<real_t>(m_particles.dx1(p))};
    const auto   j {m_particles.i2(p) + N_GHOSTS};
    const real_t dx2 {static_cast<real_t>(m_particles.dx2(p))};

    // first order
    real_t c000, c100, c010, c110, c00, c10;

    // Ex1
    // interpolate to nodes
    c000 = HALF * (m_mblock.em(i, j, em::ex1) + m_mblock.em(i - 1, j, em::ex1));
    c100 = HALF * (m_mblock.em(i, j, em::ex1) + m_mblock.em(i + 1, j, em::ex1));
    c010 = HALF * (m_mblock.em(i, j + 1, em::ex1) + m_mblock.em(i - 1, j + 1, em::ex1));
    c110 = HALF * (m_mblock.em(i, j + 1, em::ex1) + m_mblock.em(i + 1, j + 1, em::ex1));
    // interpolate from nodes to the particle position
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    e0[0] = c00 * (ONE - dx2) + c10 * dx2;
    // Ex2
    c000  = HALF * (m_mblock.em(i, j, em::ex2) + m_mblock.em(i, j - 1, em::ex2));
    c100  = HALF * (m_mblock.em(i + 1, j, em::ex2) + m_mblock.em(i + 1, j - 1, em::ex2));
    c010  = HALF * (m_mblock.em(i, j, em::ex2) + m_mblock.em(i, j + 1, em::ex2));
    c110  = HALF * (m_mblock.em(i + 1, j, em::ex2) + m_mblock.em(i + 1, j + 1, em::ex2));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    e0[1] = c00 * (ONE - dx2) + c10 * dx2;
    // Ex3
    c000  = m_mblock.em(i, j, em::ex3);
    c100  = m_mblock.em(i + 1, j, em::ex3);
    c010  = m_mblock.em(i, j + 1, em::ex3);
    c110  = m_mblock.em(i + 1, j + 1, em::ex3);
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    e0[2] = c00 * (ONE - dx2) + c10 * dx2;

    // Bx1
    c000  = HALF * (m_mblock.em(i, j, em::bx1) + m_mblock.em(i, j - 1, em::bx1));
    c100  = HALF * (m_mblock.em(i + 1, j, em::bx1) + m_mblock.em(i + 1, j - 1, em::bx1));
    c010  = HALF * (m_mblock.em(i, j, em::bx1) + m_mblock.em(i, j + 1, em::bx1));
    c110  = HALF * (m_mblock.em(i + 1, j, em::bx1) + m_mblock.em(i + 1, j + 1, em::bx1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    b0[0] = c00 * (ONE - dx2) + c10 * dx2;
    // Bx2
    c000  = HALF * (m_mblock.em(i - 1, j, em::bx2) + m_mblock.em(i, j, em::bx2));
    c100  = HALF * (m_mblock.em(i, j, em::bx2) + m_mblock.em(i + 1, j, em::bx2));
    c010  = HALF * (m_mblock.em(i - 1, j + 1, em::bx2) + m_mblock.em(i, j + 1, em::bx2));
    c110  = HALF * (m_mblock.em(i, j + 1, em::bx2) + m_mblock.em(i + 1, j + 1, em::bx2));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    b0[1] = c00 * (ONE - dx2) + c10 * dx2;
    // Bx3
    c000 = QUARTER
           * (m_mblock.em(i - 1, j - 1, em::bx3) + m_mblock.em(i - 1, j, em::bx3)
              + m_mblock.em(i, j - 1, em::bx3) + m_mblock.em(i, j, em::bx3));
    c100 = QUARTER
           * (m_mblock.em(i, j - 1, em::bx3) + m_mblock.em(i, j, em::bx3)
              + m_mblock.em(i + 1, j - 1, em::bx3) + m_mblock.em(i + 1, j, em::bx3));
    c010 = QUARTER
           * (m_mblock.em(i - 1, j, em::bx3) + m_mblock.em(i - 1, j + 1, em::bx3)
              + m_mblock.em(i, j, em::bx3) + m_mblock.em(i, j + 1, em::bx3));
    c110 = QUARTER
           * (m_mblock.em(i, j, em::bx3) + m_mblock.em(i, j + 1, em::bx3)
              + m_mblock.em(i + 1, j, em::bx3) + m_mblock.em(i + 1, j + 1, em::bx3));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    b0[2] = c00 * (ONE - dx2) + c10 * dx2;
  }

  template <>
  Inline void Pusher<Dimension::THREE_D>::interpolateFields(
    index_t& p, vec_t<Dimension::THREE_D>& e0, vec_t<Dimension::THREE_D>& b0) const {
    const auto   i {m_particles.i1(p) + N_GHOSTS};
    const real_t dx1 {static_cast<real_t>(m_particles.dx1(p))};
    const auto   j {m_particles.i2(p) + N_GHOSTS};
    const real_t dx2 {static_cast<real_t>(m_particles.dx2(p))};
    const auto   k {m_particles.i3(p) + N_GHOSTS};
    const real_t dx3 {static_cast<real_t>(m_particles.dx3(p))};

    // first order
    real_t c000, c100, c010, c110, c001, c101, c011, c111, c00, c10, c01, c11, c0, c1;

    // Ex1
    // interpolate to nodes
    c000 = HALF * (m_mblock.em(i, j, k, em::ex1) + m_mblock.em(i - 1, j, k, em::ex1));
    c100 = HALF * (m_mblock.em(i, j, k, em::ex1) + m_mblock.em(i + 1, j, k, em::ex1));
    c010 = HALF * (m_mblock.em(i, j + 1, k, em::ex1) + m_mblock.em(i - 1, j + 1, k, em::ex1));
    c110 = HALF * (m_mblock.em(i, j + 1, k, em::ex1) + m_mblock.em(i + 1, j + 1, k, em::ex1));
    // interpolate from nodes to the particle position
    c00 = c000 * (ONE - dx1) + c100 * dx1;
    c10 = c010 * (ONE - dx1) + c110 * dx1;
    c0  = c00 * (ONE - dx2) + c10 * dx2;
    // interpolate to nodes
    c001 = HALF * (m_mblock.em(i, j, k + 1, em::ex1) + m_mblock.em(i - 1, j, k + 1, em::ex1));
    c101 = HALF * (m_mblock.em(i, j, k + 1, em::ex1) + m_mblock.em(i + 1, j, k + 1, em::ex1));
    c011
      = HALF
        * (m_mblock.em(i, j + 1, k + 1, em::ex1) + m_mblock.em(i - 1, j + 1, k + 1, em::ex1));
    c111
      = HALF
        * (m_mblock.em(i, j + 1, k + 1, em::ex1) + m_mblock.em(i + 1, j + 1, k + 1, em::ex1));
    // interpolate from nodes to the particle position
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    e0[0] = c0 * (ONE - dx3) + c1 * dx3;

    // Ex2
    c000 = HALF * (m_mblock.em(i, j, k, em::ex2) + m_mblock.em(i, j - 1, k, em::ex2));
    c100 = HALF * (m_mblock.em(i + 1, j, k, em::ex2) + m_mblock.em(i + 1, j - 1, k, em::ex2));
    c010 = HALF * (m_mblock.em(i, j, k, em::ex2) + m_mblock.em(i, j + 1, k, em::ex2));
    c110 = HALF * (m_mblock.em(i + 1, j, k, em::ex2) + m_mblock.em(i + 1, j + 1, k, em::ex2));
    c00  = c000 * (ONE - dx1) + c100 * dx1;
    c10  = c010 * (ONE - dx1) + c110 * dx1;
    c0   = c00 * (ONE - dx2) + c10 * dx2;
    c001 = HALF * (m_mblock.em(i, j, k + 1, em::ex2) + m_mblock.em(i, j - 1, k + 1, em::ex2));
    c101
      = HALF
        * (m_mblock.em(i + 1, j, k + 1, em::ex2) + m_mblock.em(i + 1, j - 1, k + 1, em::ex2));
    c011 = HALF * (m_mblock.em(i, j, k + 1, em::ex2) + m_mblock.em(i, j + 1, k + 1, em::ex2));
    c111
      = HALF
        * (m_mblock.em(i + 1, j, k + 1, em::ex2) + m_mblock.em(i + 1, j + 1, k + 1, em::ex2));
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    e0[1] = c0 * (ONE - dx3) + c1 * dx3;

    // Ex3
    c000 = HALF * (m_mblock.em(i, j, k, em::ex3) + m_mblock.em(i, j, k - 1, em::ex3));
    c100 = HALF * (m_mblock.em(i + 1, j, k, em::ex3) + m_mblock.em(i + 1, j, k - 1, em::ex3));
    c010 = HALF * (m_mblock.em(i, j + 1, k, em::ex3) + m_mblock.em(i, j + 1, k - 1, em::ex3));
    c110
      = HALF
        * (m_mblock.em(i + 1, j + 1, k, em::ex3) + m_mblock.em(i + 1, j + 1, k - 1, em::ex3));
    c001 = HALF * (m_mblock.em(i, j, k, em::ex3) + m_mblock.em(i, j, k + 1, em::ex3));
    c101 = HALF * (m_mblock.em(i + 1, j, k, em::ex3) + m_mblock.em(i + 1, j, k + 1, em::ex3));
    c011 = HALF * (m_mblock.em(i, j + 1, k, em::ex3) + m_mblock.em(i, j + 1, k + 1, em::ex3));
    c111
      = HALF
        * (m_mblock.em(i + 1, j + 1, k, em::ex3) + m_mblock.em(i + 1, j + 1, k + 1, em::ex3));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    e0[2] = c0 * (ONE - dx3) + c1 * dx3;

    // Bx1
    c000 = QUARTER
           * (m_mblock.em(i, j, k, em::bx1) + m_mblock.em(i, j - 1, k, em::bx1)
              + m_mblock.em(i, j, k - 1, em::bx1) + m_mblock.em(i, j - 1, k - 1, em::bx1));
    c100 = QUARTER
           * (m_mblock.em(i + 1, j, k, em::bx1) + m_mblock.em(i + 1, j - 1, k, em::bx1)
              + m_mblock.em(i + 1, j, k - 1, em::bx1)
              + m_mblock.em(i + 1, j - 1, k - 1, em::bx1));
    c001 = QUARTER
           * (m_mblock.em(i, j, k, em::bx1) + m_mblock.em(i, j, k + 1, em::bx1)
              + m_mblock.em(i, j - 1, k, em::bx1) + m_mblock.em(i, j - 1, k + 1, em::bx1));
    c101 = QUARTER
           * (m_mblock.em(i + 1, j, k, em::bx1) + m_mblock.em(i + 1, j, k + 1, em::bx1)
              + m_mblock.em(i + 1, j - 1, k, em::bx1)
              + m_mblock.em(i + 1, j - 1, k + 1, em::bx1));
    c010 = QUARTER
           * (m_mblock.em(i, j, k, em::bx1) + m_mblock.em(i, j + 1, k, em::bx1)
              + m_mblock.em(i, j, k - 1, em::bx1) + m_mblock.em(i, j + 1, k - 1, em::bx1));
    c110 = QUARTER
           * (m_mblock.em(i + 1, j, k, em::bx1) + m_mblock.em(i + 1, j, k - 1, em::bx1)
              + m_mblock.em(i + 1, j + 1, k - 1, em::bx1)
              + m_mblock.em(i + 1, j + 1, k, em::bx1));
    c011 = QUARTER
           * (m_mblock.em(i, j, k, em::bx1) + m_mblock.em(i, j + 1, k, em::bx1)
              + m_mblock.em(i, j + 1, k + 1, em::bx1) + m_mblock.em(i, j, k + 1, em::bx1));
    c111 = QUARTER
           * (m_mblock.em(i + 1, j, k, em::bx1) + m_mblock.em(i + 1, j + 1, k, em::bx1)
              + m_mblock.em(i + 1, j + 1, k + 1, em::bx1)
              + m_mblock.em(i + 1, j, k + 1, em::bx1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    b0[0] = c0 * (ONE - dx3) + c1 * dx3;

    // Bx2
    c000 = QUARTER
           * (m_mblock.em(i - 1, j, k - 1, em::bx2) + m_mblock.em(i - 1, j, k, em::bx2)
              + m_mblock.em(i, j, k - 1, em::bx2) + m_mblock.em(i, j, k, em::bx2));
    c100 = QUARTER
           * (m_mblock.em(i, j, k - 1, em::bx2) + m_mblock.em(i, j, k, em::bx2)
              + m_mblock.em(i + 1, j, k - 1, em::bx2) + m_mblock.em(i + 1, j, k, em::bx2));
    c001 = QUARTER
           * (m_mblock.em(i - 1, j, k, em::bx2) + m_mblock.em(i - 1, j, k + 1, em::bx2)
              + m_mblock.em(i, j, k, em::bx2) + m_mblock.em(i, j, k + 1, em::bx2));
    c101 = QUARTER
           * (m_mblock.em(i, j, k, em::bx2) + m_mblock.em(i, j, k + 1, em::bx2)
              + m_mblock.em(i + 1, j, k, em::bx2) + m_mblock.em(i + 1, j, k + 1, em::bx2));
    c010 = QUARTER
           * (m_mblock.em(i - 1, j + 1, k - 1, em::bx2) + m_mblock.em(i - 1, j + 1, k, em::bx2)
              + m_mblock.em(i, j + 1, k - 1, em::bx2) + m_mblock.em(i, j + 1, k, em::bx2));
    c110 = QUARTER
           * (m_mblock.em(i, j + 1, k - 1, em::bx2) + m_mblock.em(i, j + 1, k, em::bx2)
              + m_mblock.em(i + 1, j + 1, k - 1, em::bx2)
              + m_mblock.em(i + 1, j + 1, k, em::bx2));
    c011 = QUARTER
           * (m_mblock.em(i - 1, j + 1, k, em::bx2) + m_mblock.em(i - 1, j + 1, k + 1, em::bx2)
              + m_mblock.em(i, j + 1, k, em::bx2) + m_mblock.em(i, j + 1, k + 1, em::bx2));
    c111 = QUARTER
           * (m_mblock.em(i, j + 1, k, em::bx2) + m_mblock.em(i, j + 1, k + 1, em::bx2)
              + m_mblock.em(i + 1, j + 1, k, em::bx2)
              + m_mblock.em(i + 1, j + 1, k + 1, em::bx2));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    b0[1] = c0 * (ONE - dx3) + c1 * dx3;

    // Bx3
    c000 = QUARTER
           * (m_mblock.em(i - 1, j - 1, k, em::bx3) + m_mblock.em(i - 1, j, k, em::bx3)
              + m_mblock.em(i, j - 1, k, em::bx3) + m_mblock.em(i, j, k, em::bx3));
    c100 = QUARTER
           * (m_mblock.em(i, j - 1, k, em::bx3) + m_mblock.em(i, j, k, em::bx3)
              + m_mblock.em(i + 1, j - 1, k, em::bx3) + m_mblock.em(i + 1, j, k, em::bx3));
    c001 = QUARTER
           * (m_mblock.em(i - 1, j - 1, k + 1, em::bx3) + m_mblock.em(i - 1, j, k + 1, em::bx3)
              + m_mblock.em(i, j - 1, k + 1, em::bx3) + m_mblock.em(i, j, k + 1, em::bx3));
    c101 = QUARTER
           * (m_mblock.em(i, j - 1, k + 1, em::bx3) + m_mblock.em(i, j, k + 1, em::bx3)
              + m_mblock.em(i + 1, j - 1, k + 1, em::bx3)
              + m_mblock.em(i + 1, j, k + 1, em::bx3));
    c010 = QUARTER
           * (m_mblock.em(i - 1, j, k, em::bx3) + m_mblock.em(i - 1, j + 1, k, em::bx3)
              + m_mblock.em(i, j, k, em::bx3) + m_mblock.em(i, j + 1, k, em::bx3));
    c110 = QUARTER
           * (m_mblock.em(i, j, k, em::bx3) + m_mblock.em(i, j + 1, k, em::bx3)
              + m_mblock.em(i + 1, j, k, em::bx3) + m_mblock.em(i + 1, j + 1, k, em::bx3));
    c011 = QUARTER
           * (m_mblock.em(i - 1, j, k + 1, em::bx3) + m_mblock.em(i - 1, j + 1, k + 1, em::bx3)
              + m_mblock.em(i, j, k + 1, em::bx3) + m_mblock.em(i, j + 1, k + 1, em::bx3));
    c111 = QUARTER
           * (m_mblock.em(i, j, k + 1, em::bx3) + m_mblock.em(i, j + 1, k + 1, em::bx3)
              + m_mblock.em(i + 1, j, k + 1, em::bx3)
              + m_mblock.em(i + 1, j + 1, k + 1, em::bx3));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    b0[2] = c0 * (ONE - dx3) + c1 * dx3;
  }

} // namespace ntt

#endif