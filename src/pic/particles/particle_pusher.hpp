#ifndef PIC_PARTICLE_PUSHER_H
#define PIC_PARTICLE_PUSHER_H

#include "global.h"
#include "fields.h"
#include "particles.h"
#include "meshblock.h"
#include "pic.h"

#include "field_macros.h"
#include "particle_macros.h"

#include <stdexcept>
#include <iostream>

namespace ntt {
  struct Boris_t {};
  struct Photon_t {};

  /**
   * @brief Algorithm for the Particle pusher.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Pusher_kernel {
    Meshblock<D, TypePIC> m_mblock;
    Particles<D, TypePIC> m_particles;
    real_t                m_coeff, m_dt;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     * @param particles Particles.
     * @param coeff Coefficient to be multiplied by dE/dt = coeff * curl B.
     * @param dt Time step.
     */
    Pusher_kernel(const Meshblock<D, TypePIC>& mblock,
                  const Particles<D, TypePIC>& particles,
                  const real_t&                coeff,
                  const real_t&                dt)
      : m_mblock(mblock), m_particles(particles), m_coeff(coeff), m_dt(dt) {}
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
        // push boris-particles
        auto range_policy
          = Kokkos::RangePolicy<AccelExeSpace, Boris_t>(0, m_particles.npart());
        Kokkos::parallel_for("pusher", range_policy, *this);
      } else {
        NTTHostError("pusher not implemented");
      }
    }
    /**
     * @brief Pusher for the forward Boris algorithm.
     * @param p index.
     */
    Inline void operator()(const Boris_t&, index_t p) const {
      if (!m_particles.is_dead(p)) {

        vec_t<Dim3> e_int, b_int, e_int_Cart, b_int_Cart;
        interpolateFields(p, e_int, b_int);

#ifdef MINKOWSKI_METRIC
        coord_t<D> xp;
#else
        coord_t<Dim3> xp;
#endif
        getParticleCoordinate(p, xp);
        m_mblock.metric.v_Cntrv2Cart(xp, e_int, e_int_Cart);
        m_mblock.metric.v_Cntrv2Cart(xp, b_int, b_int_Cart);

        BorisUpdate(p, e_int_Cart, b_int_Cart);

        real_t inv_energy;
        inv_energy = ONE / get_prtl_Gamma_SR(m_particles, p);

        // contravariant 3-velocity: u^i / gamma
        vec_t<Dim3> v;
        m_mblock.metric.v_Cart2Cntrv(
          xp, {m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p)}, v);
        v[0] *= inv_energy;
        v[1] *= inv_energy;
        v[2] *= inv_energy;

        positionUpdate(p, v);
      }
    }
    /**
     * @brief Pusher for the photon.
     * @param p index.
     */
    Inline void operator()(const Photon_t&, index_t p) const {
      if (!m_particles.is_dead(p)) {

#ifdef MINKOWSKI_METRIC
        coord_t<D> xp;
#else
        coord_t<Dim3> xp;
#endif
        getParticleCoordinate(p, xp);
        vec_t<Dim3> v;
        m_mblock.metric.v_Cart2Cntrv(
          xp, {m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p)}, v);

        real_t inv_energy;
        inv_energy = ONE / math::sqrt(get_prtl_Usqr_SR(m_particles, p));
        v[0] *= inv_energy;
        v[1] *= inv_energy;
        v[2] *= inv_energy;

        positionUpdate(p, v);
      }
    }

#ifdef MINKOWSKI_METRIC
    /**
     * @brief Transform particle coordinate from code units i+di to `real_t` type.
     * @param p index of the particle.
     * @param coord coordinate of the particle as a vector (of size D).
     */
    Inline void getParticleCoordinate(index_t&, coord_t<D>&) const;
#else
    /**
     * @brief Transform particle coordinate from code units i+di to `real_t` type.
     * @param p index of the particle.
     * @param coord coordinate of the particle as a vector (of size 3).
     */
    Inline void getParticleCoordinate(index_t&, coord_t<Dim3>&) const;
#endif

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
  };

#ifdef MINKOWSKI_METRIC
  template <>
  Inline void Pusher_kernel<Dim1>::getParticleCoordinate(index_t& p, coord_t<Dim1>& xp) const {
    xp[0] = get_prtl_x1(m_particles, p);
  }
  template <>
  Inline void Pusher_kernel<Dim2>::getParticleCoordinate(index_t& p, coord_t<Dim2>& xp) const {
    xp[0] = get_prtl_x1(m_particles, p);
    xp[1] = get_prtl_x2(m_particles, p);
  }
#else
  template <>
  Inline void Pusher_kernel<Dim1>::getParticleCoordinate(index_t&, coord_t<Dim3>&) const {
    NTTError("not implemented");
  }
  template <>
  Inline void Pusher_kernel<Dim2>::getParticleCoordinate(index_t& p, coord_t<Dim3>& xp) const {
    xp[0] = get_prtl_x1(m_particles, p);
    xp[1] = get_prtl_x2(m_particles, p);
    xp[2] = m_particles.phi(p);
  }
#endif

  template <>
  Inline void Pusher_kernel<Dim3>::getParticleCoordinate(index_t& p, coord_t<Dim3>& xp) const {
    xp[0] = get_prtl_x1(m_particles, p);
    xp[1] = get_prtl_x2(m_particles, p);
    xp[2] = get_prtl_x3(m_particles, p);
  }

  // * * * * * * * * * * * * * * *
  // General position update
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Pusher_kernel<Dim1>::positionUpdate(index_t& p, const vec_t<Dim3>& v) const {
    positionUpdate_x1(p, v[0]);
  }
  template <>
  Inline void Pusher_kernel<Dim2>::positionUpdate(index_t& p, const vec_t<Dim3>& v) const {
    positionUpdate_x1(p, v[0]);
    positionUpdate_x2(p, v[1]);
#ifndef MINKOWSKI_METRIC
    m_particles.phi(p) += m_dt * v[2];
#endif
  }
  template <>
  Inline void Pusher_kernel<Dim3>::positionUpdate(index_t& p, const vec_t<Dim3>& v) const {
    positionUpdate_x1(p, v[0]);
    positionUpdate_x2(p, v[1]);
    positionUpdate_x3(p, v[2]);
  }

  template <Dimension D>
  Inline void Pusher_kernel<D>::positionUpdate_x1(index_t& p, const real_t& vx1) const {
    m_particles.dx1(p) = m_particles.dx1(p) + static_cast<float>(m_dt * vx1);
    int   temp_i {static_cast<int>(m_particles.dx1(p))};
    float temp_r {math::fmax(SIGNf(m_particles.dx1(p)) + temp_i, static_cast<float>(temp_i))
                  - 1.0f};
    temp_i             = static_cast<int>(temp_r);
    m_particles.i1(p)  = m_particles.i1(p) + temp_i;
    m_particles.dx1(p) = m_particles.dx1(p) - temp_r;
  }
  template <Dimension D>
  Inline void Pusher_kernel<D>::positionUpdate_x2(index_t& p, const real_t& vx2) const {
    m_particles.dx2(p) = m_particles.dx2(p) + static_cast<float>(m_dt * vx2);
    int   temp_i {static_cast<int>(m_particles.dx2(p))};
    float temp_r {math::fmax(SIGNf(m_particles.dx2(p)) + temp_i, static_cast<float>(temp_i))
                  - 1.0f};
    temp_i             = static_cast<int>(temp_r);
    m_particles.i2(p)  = m_particles.i2(p) + temp_i;
    m_particles.dx2(p) = m_particles.dx2(p) - temp_r;
  }
  template <Dimension D>
  Inline void Pusher_kernel<D>::positionUpdate_x3(index_t& p, const real_t& vx3) const {
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
  Inline void
  Pusher_kernel<D>::BorisUpdate(index_t& p, vec_t<Dim3>& e0, vec_t<Dim3>& b0) const {
    real_t COEFF {m_coeff};

    e0[0] *= COEFF;
    e0[1] *= COEFF;
    e0[2] *= COEFF;
    vec_t<Dim3> u0 {
      m_particles.ux1(p) + e0[0], m_particles.ux2(p) + e0[1], m_particles.ux3(p) + e0[2]};

    COEFF *= ONE / math::sqrt(ONE + SQR(u0[0]) + SQR(u0[1]) + SQR(u0[2]));
    b0[0] *= COEFF;
    b0[1] *= COEFF;
    b0[2] *= COEFF;
    COEFF = 2.0 / (ONE + SQR(b0[0]) + SQR(b0[1]) + SQR(b0[2]));

    vec_t<Dim3> u1 {(u0[0] + u0[1] * b0[2] - u0[2] * b0[1]) * COEFF,
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
  Inline void
  Pusher_kernel<Dim1>::interpolateFields(index_t& p, vec_t<Dim3>& e0, vec_t<Dim3>& b0) const {
    const auto   i {m_particles.i1(p) + N_GHOSTS};
    const real_t dx1 {static_cast<real_t>(m_particles.dx1(p))};

    // first order
    real_t c0, c1;

    // Ex1
    // interpolate to nodes
    c0 = HALF * (EX1(i) + EX1(i - 1));
    c1 = HALF * (EX1(i) + EX1(i + 1));
    // interpolate from nodes to the particle position
    e0[0] = c0 * (ONE - dx1) + c1 * dx1;
    // Ex2
    c0    = EX2(i);
    c1    = EX2(i + 1);
    e0[1] = c0 * (ONE - dx1) + c1 * dx1;
    // Ex3
    c0    = EX3(i);
    c1    = EX3(i + 1);
    e0[2] = c0 * (ONE - dx1) + c1 * dx1;

    // Bx1
    c0    = BX1(i);
    c1    = BX1(i + 1);
    b0[0] = c0 * (ONE - dx1) + c1 * dx1;
    // Bx2
    c0    = HALF * (BX2(i - 1) + BX2(i));
    c1    = HALF * (BX2(i) + BX2(i + 1));
    b0[1] = c0 * (ONE - dx1) + c1 * dx1;
    // Bx3
    c0    = HALF * (BX3(i - 1) + BX3(i));
    c1    = HALF * (BX3(i) + BX3(i + 1));
    b0[2] = c0 * (ONE - dx1) + c1 * dx1;
  }

  template <>
  Inline void
  Pusher_kernel<Dim2>::interpolateFields(index_t& p, vec_t<Dim3>& e0, vec_t<Dim3>& b0) const {
    const auto   i {m_particles.i1(p) + N_GHOSTS};
    const real_t dx1 {static_cast<real_t>(m_particles.dx1(p))};
    const auto   j {m_particles.i2(p) + N_GHOSTS};
    const real_t dx2 {static_cast<real_t>(m_particles.dx2(p))};

    // first order
    real_t c000, c100, c010, c110, c00, c10;

    // Ex1
    // interpolate to nodes
    c000 = HALF * (EX1(i, j) + EX1(i - 1, j));
    c100 = HALF * (EX1(i, j) + EX1(i + 1, j));
    c010 = HALF * (EX1(i, j + 1) + EX1(i - 1, j + 1));
    c110 = HALF * (EX1(i, j + 1) + EX1(i + 1, j + 1));
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
  Inline void
  Pusher_kernel<Dim3>::interpolateFields(index_t& p, vec_t<Dim3>& e0, vec_t<Dim3>& b0) const {
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
    c000 = HALF * (EX1(i, j, k) + EX1(i - 1, j, k));
    c100 = HALF * (EX1(i, j, k) + EX1(i + 1, j, k));
    c010 = HALF * (EX1(i, j + 1, k) + EX1(i - 1, j + 1, k));
    c110 = HALF * (EX1(i, j + 1, k) + EX1(i + 1, j + 1, k));
    // interpolate from nodes to the particle position
    c00 = c000 * (ONE - dx1) + c100 * dx1;
    c10 = c010 * (ONE - dx1) + c110 * dx1;
    c0  = c00 * (ONE - dx2) + c10 * dx2;
    // interpolate to nodes
    c001 = HALF * (EX1(i, j, k + 1) + EX1(i - 1, j, k + 1));
    c101 = HALF * (EX1(i, j, k + 1) + EX1(i + 1, j, k + 1));
    c011 = HALF * (EX1(i, j + 1, k + 1) + EX1(i - 1, j + 1, k + 1));
    c111 = HALF * (EX1(i, j + 1, k + 1) + EX1(i + 1, j + 1, k + 1));
    // interpolate from nodes to the particle position
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    e0[0] = c0 * (ONE - dx3) + c1 * dx3;

    // Ex2
    c000  = HALF * (EX2(i, j, k) + EX2(i, j - 1, k));
    c100  = HALF * (EX2(i + 1, j, k) + EX2(i + 1, j - 1, k));
    c010  = HALF * (EX2(i, j, k) + EX2(i, j + 1, k));
    c110  = HALF * (EX2(i + 1, j, k) + EX2(i + 1, j + 1, k));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    c001  = HALF * (EX2(i, j, k + 1) + EX2(i, j - 1, k + 1));
    c101  = HALF * (EX2(i + 1, j, k + 1) + EX2(i + 1, j - 1, k + 1));
    c011  = HALF * (EX2(i, j, k + 1) + EX2(i, j + 1, k + 1));
    c111  = HALF * (EX2(i + 1, j, k + 1) + EX2(i + 1, j + 1, k + 1));
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    e0[1] = c0 * (ONE - dx3) + c1 * dx3;

    // Ex3
    c000  = HALF * (EX3(i, j, k) + EX3(i, j, k - 1));
    c100  = HALF * (EX3(i + 1, j, k) + EX3(i + 1, j, k - 1));
    c010  = HALF * (EX3(i, j + 1, k) + EX3(i, j + 1, k - 1));
    c110  = HALF * (EX3(i + 1, j + 1, k) + EX3(i + 1, j + 1, k - 1));
    c001  = HALF * (EX3(i, j, k) + EX3(i, j, k + 1));
    c101  = HALF * (EX3(i + 1, j, k) + EX3(i + 1, j, k + 1));
    c011  = HALF * (EX3(i, j + 1, k) + EX3(i, j + 1, k + 1));
    c111  = HALF * (EX3(i + 1, j + 1, k) + EX3(i + 1, j + 1, k + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    e0[2] = c0 * (ONE - dx3) + c1 * dx3;

    // Bx1
    c000 = INV_4 * (BX1(i, j, k) + BX1(i, j - 1, k) + BX1(i, j, k - 1) + BX1(i, j - 1, k - 1));
    c100 = INV_4
           * (BX1(i + 1, j, k) + BX1(i + 1, j - 1, k) + BX1(i + 1, j, k - 1)
              + BX1(i + 1, j - 1, k - 1));
    c001 = INV_4 * (BX1(i, j, k) + BX1(i, j, k + 1) + BX1(i, j - 1, k) + BX1(i, j - 1, k + 1));
    c101 = INV_4
           * (BX1(i + 1, j, k) + BX1(i + 1, j, k + 1) + BX1(i + 1, j - 1, k)
              + BX1(i + 1, j - 1, k + 1));
    c010 = INV_4 * (BX1(i, j, k) + BX1(i, j + 1, k) + BX1(i, j, k - 1) + BX1(i, j + 1, k - 1));
    c110 = INV_4
           * (BX1(i + 1, j, k) + BX1(i + 1, j, k - 1) + BX1(i + 1, j + 1, k - 1)
              + BX1(i + 1, j + 1, k));
    c011 = INV_4 * (BX1(i, j, k) + BX1(i, j + 1, k) + BX1(i, j + 1, k + 1) + BX1(i, j, k + 1));
    c111 = INV_4
           * (BX1(i + 1, j, k) + BX1(i + 1, j + 1, k) + BX1(i + 1, j + 1, k + 1)
              + BX1(i + 1, j, k + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    b0[0] = c0 * (ONE - dx3) + c1 * dx3;

    // Bx2
    c000 = INV_4 * (BX2(i - 1, j, k - 1) + BX2(i - 1, j, k) + BX2(i, j, k - 1) + BX2(i, j, k));
    c100 = INV_4 * (BX2(i, j, k - 1) + BX2(i, j, k) + BX2(i + 1, j, k - 1) + BX2(i + 1, j, k));
    c001 = INV_4 * (BX2(i - 1, j, k) + BX2(i - 1, j, k + 1) + BX2(i, j, k) + BX2(i, j, k + 1));
    c101 = INV_4 * (BX2(i, j, k) + BX2(i, j, k + 1) + BX2(i + 1, j, k) + BX2(i + 1, j, k + 1));
    c010 = INV_4
           * (BX2(i - 1, j + 1, k - 1) + BX2(i - 1, j + 1, k) + BX2(i, j + 1, k - 1)
              + BX2(i, j + 1, k));
    c110 = INV_4
           * (BX2(i, j + 1, k - 1) + BX2(i, j + 1, k) + BX2(i + 1, j + 1, k - 1)
              + BX2(i + 1, j + 1, k));
    c011 = INV_4
           * (BX2(i - 1, j + 1, k) + BX2(i - 1, j + 1, k + 1) + BX2(i, j + 1, k)
              + BX2(i, j + 1, k + 1));
    c111 = INV_4
           * (BX2(i, j + 1, k) + BX2(i, j + 1, k + 1) + BX2(i + 1, j + 1, k)
              + BX2(i + 1, j + 1, k + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    b0[1] = c0 * (ONE - dx3) + c1 * dx3;

    // Bx3
    c000 = INV_4 * (BX3(i - 1, j - 1, k) + BX3(i - 1, j, k) + BX3(i, j - 1, k) + BX3(i, j, k));
    c100 = INV_4 * (BX3(i, j - 1, k) + BX3(i, j, k) + BX3(i + 1, j - 1, k) + BX3(i + 1, j, k));
    c001 = INV_4
           * (BX3(i - 1, j - 1, k + 1) + BX3(i - 1, j, k + 1) + BX3(i, j - 1, k + 1)
              + BX3(i, j, k + 1));
    c101 = INV_4
           * (BX3(i, j - 1, k + 1) + BX3(i, j, k + 1) + BX3(i + 1, j - 1, k + 1)
              + BX3(i + 1, j, k + 1));
    c010 = INV_4 * (BX3(i - 1, j, k) + BX3(i - 1, j + 1, k) + BX3(i, j, k) + BX3(i, j + 1, k));
    c110 = INV_4 * (BX3(i, j, k) + BX3(i, j + 1, k) + BX3(i + 1, j, k) + BX3(i + 1, j + 1, k));
    c011 = INV_4
           * (BX3(i - 1, j, k + 1) + BX3(i - 1, j + 1, k + 1) + BX3(i, j, k + 1)
              + BX3(i, j + 1, k + 1));
    c111 = INV_4
           * (BX3(i, j, k + 1) + BX3(i, j + 1, k + 1) + BX3(i + 1, j, k + 1)
              + BX3(i + 1, j + 1, k + 1));
    c00   = c000 * (ONE - dx1) + c100 * dx1;
    c01   = c001 * (ONE - dx1) + c101 * dx1;
    c10   = c010 * (ONE - dx1) + c110 * dx1;
    c11   = c011 * (ONE - dx1) + c111 * dx1;
    c0    = c00 * (ONE - dx2) + c10 * dx2;
    c1    = c01 * (ONE - dx2) + c11 * dx2;
    b0[2] = c0 * (ONE - dx3) + c1 * dx3;
  }

} // namespace ntt

// Inline void operator()(const BorisBwd_t&, index_t p) const {
//   real_t inv_energy;
//   inv_energy = SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p)) +
//   SQR(m_particles.ux3(p)); inv_energy = ONE / math::sqrt(ONE + inv_energy);

//   coord_t<D> xp;
//   getParticleCoordinate(p, xp);

//   vec_t<Dim3> v;
//   m_mblock.metric.v_Cart2Cntrv(
//     xp, {m_particles.ux1(p), m_particles.ux2(p), m_particles.ux3(p)}, v);
//   v[0] *= inv_energy;
//   v[1] *= inv_energy;
//   v[2] *= inv_energy;
//   positionUpdate(p, v);
//   getParticleCoordinate(p, xp);

//   vec_t<Dim3> e_int, b_int, e_int_Cart, b_int_Cart;
//   interpolateFields(p, e_int, b_int);

//   m_mblock.metric.v_Cntrv2Cart(xp, e_int, e_int_Cart);
//   m_mblock.metric.v_Cntrv2Cart(xp, b_int, b_int_Cart);

//   BorisUpdate(p, e_int_Cart, b_int_Cart);
// }

#endif