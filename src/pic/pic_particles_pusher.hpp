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
  struct PhotonFwd_t {};
  struct BorisBwd_t {};
  struct PhotonBwd_t {};

  /**
   * Algorithm for the Particle pusher.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Pusher {
    using index_t = const std::size_t;
    Meshblock<D, SimulationType::PIC> m_mblock;
    Particles<D, SimulationType::PIC> m_particles;
    real_t m_coeff, m_dt;

  public:
    Pusher(const Meshblock<D, SimulationType::PIC>& mblock,
           const Particles<D, SimulationType::PIC>& particles,
           const real_t& coeff,
           const real_t& dt)
      : m_mblock(mblock), m_particles(particles), m_coeff(coeff), m_dt(dt) {}
    /**
     * Loop over all active particles of the given species and call the appropriate pusher.
     *
     */
    void pushParticles() {
      if (m_coeff > ZERO) {
        if (m_particles.pusher() == ParticlePusher::PHOTON) {
          // push photons forward
          auto range_policy = Kokkos::RangePolicy<AccelExeSpace, PhotonFwd_t>(0, m_particles.npart());
          Kokkos::parallel_for("pusher", range_policy, *this);
        } else if (m_particles.pusher() == ParticlePusher::BORIS) {
          // push boris-particles forward
          auto range_policy = Kokkos::RangePolicy<AccelExeSpace, BorisFwd_t>(0, m_particles.npart());
          Kokkos::parallel_for("pusher", range_policy, *this);
        } else {
          NTTError("pusher not implemented");
        }
      } else {
        if (m_particles.pusher() == ParticlePusher::PHOTON) {
          // push photons backward
          auto range_policy = Kokkos::RangePolicy<AccelExeSpace, PhotonBwd_t>(0, m_particles.npart());
          Kokkos::parallel_for("pusher", range_policy, *this);
        } else if (m_particles.pusher() == ParticlePusher::BORIS) {
          // push boris-particles backward
          auto range_policy = Kokkos::RangePolicy<AccelExeSpace, BorisBwd_t>(0, m_particles.npart());
          Kokkos::parallel_for("pusher", range_policy, *this);
        } else {
          NTTError("pusher not implemented");
        }
      }
    }
    /**
     * @todo Faster sqrt method?
     */
    Inline void operator()(const BorisFwd_t&, const index_t p) const {
      real_t e0_x1, e0_x2, e0_x3;
      real_t b0_x1, b0_x2, b0_x3;
      interpolateFields(p, e0_x1, e0_x2, e0_x3, b0_x1, b0_x2, b0_x3);
      // convertToCartesian(p);
      BorisUpdate(p, e0_x1, e0_x2, e0_x3, b0_x1, b0_x2, b0_x3);
      real_t inv_gamma0 {
        ONE / std::sqrt(ONE + SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p)) + SQR(m_particles.ux3(p)))};
      positionUpdate(p, inv_gamma0);
      // convertFromCartesian(p);
    }
    /**
     * @todo Faster sqrt method?
     */
    Inline void operator()(const PhotonFwd_t&, const index_t p) const {
      real_t inv_energy {
        ONE / std::sqrt(SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p)) + SQR(m_particles.ux3(p)))};
      positionUpdate(p, inv_energy);
    }

    Inline void operator()(const BorisBwd_t&, const index_t p) const {
      real_t inv_gamma0 {
        ONE / std::sqrt(ONE + SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p)) + SQR(m_particles.ux3(p)))};
      positionUpdate(p, inv_gamma0);
      real_t e0_x1, e0_x2, e0_x3;
      real_t b0_x1, b0_x2, b0_x3;
      interpolateFields(p, e0_x1, e0_x2, e0_x3, b0_x1, b0_x2, b0_x3);
      // convertToCartesian(p);
      BorisUpdate(p, e0_x1, e0_x2, e0_x3, b0_x1, b0_x2, b0_x3);
      // convertFromCartesian(p);
    }
    Inline void operator()(const PhotonBwd_t&, const index_t p) const {
      real_t inv_energy {ONE / std::sqrt(SQR(m_particles.ux1(p)) + SQR(m_particles.ux2(p)) + SQR(m_particles.ux3(p)))};
      positionUpdate(p, inv_energy);
    }

    /**
     * First order Yee mesh field interpolation to particle position.
     *
     * @param p index of the particle.
     * @param eb interpolated field components.
     */
    Inline void interpolateFields(const index_t&, real_t&, real_t&, real_t&, real_t&, real_t&, real_t&) const;

    /**
     * Update particle positions according to updated velocities.
     *
     * @param p index of the particle.
     * @param inv_energy inverse of energy (to compute 3-velocity).
     */
    Inline void positionUpdate(const index_t&, const real_t&) const;

    /**
     * Update each position component.
     *
     * @param p index of the particle.
     * @param inv_energy inverse of energy (to compute 3 velocity).
     */
    Inline void positionUpdate_x1(const index_t&, const real_t&) const;
    Inline void positionUpdate_x2(const index_t&, const real_t&) const;
    Inline void positionUpdate_x3(const index_t&, const real_t&) const;

    /**
     * Boris algorithm.
     * @note Fields are modified inside the function and cannot be reused.
     *
     * @param p index of the particle.
     * @param eb interpolated field components.
     */
    Inline void BorisUpdate(const index_t&, real_t&, real_t&, real_t&, real_t&, real_t&, real_t&) const;
  };

  // * * * * * * * * * * * * * * *
  // General position update
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Pusher<Dimension::ONE_D>::positionUpdate(const index_t& p, const real_t& inv_energy) const {
    positionUpdate_x1(p, inv_energy);
  }
  template <>
  Inline void Pusher<Dimension::TWO_D>::positionUpdate(const index_t& p, const real_t& inv_energy) const {
    positionUpdate_x1(p, inv_energy);
    positionUpdate_x2(p, inv_energy);
  }
  template <>
  Inline void Pusher<Dimension::THREE_D>::positionUpdate(const index_t& p, const real_t& inv_energy) const {
    positionUpdate_x1(p, inv_energy);
    positionUpdate_x2(p, inv_energy);
    positionUpdate_x3(p, inv_energy);
  }

  template <Dimension D>
  Inline void Pusher<D>::positionUpdate_x1(const index_t& p, const real_t& inv_energy) const {
    m_particles.dx1(p) = m_particles.dx1(p) + static_cast<float>(m_dt * m_particles.ux1(p) * inv_energy);
    int temp_i {static_cast<int>(m_particles.dx1(p))};
    float temp_r {std::max(SIGNf(m_particles.dx1(p)) + temp_i, static_cast<float>(temp_i)) - 1.0f};
    temp_i = static_cast<int>(temp_r);
    m_particles.i1(p) = m_particles.i1(p) + temp_i;
    m_particles.dx1(p) = m_particles.dx1(p) - temp_r;
  }
  template <Dimension D>
  Inline void Pusher<D>::positionUpdate_x2(const index_t& p, const real_t& inv_energy) const {
    m_particles.dx2(p) = m_particles.dx2(p) + static_cast<float>(m_dt * m_particles.ux2(p) * inv_energy);
    int temp_i {static_cast<int>(m_particles.dx2(p))};
    float temp_r {std::max(SIGNf(m_particles.dx2(p)) + temp_i, static_cast<float>(temp_i)) - 1.0f};
    temp_i = static_cast<int>(temp_r);
    m_particles.i2(p) = m_particles.i2(p) + temp_i;
    m_particles.dx2(p) = m_particles.dx2(p) - temp_r;
  }
  template <Dimension D>
  Inline void Pusher<D>::positionUpdate_x3(const index_t& p, const real_t& inv_energy) const {
    m_particles.dx3(p) = m_particles.dx3(p) + static_cast<float>(m_dt * m_particles.ux3(p) * inv_energy);
    int temp_i {static_cast<int>(m_particles.dx3(p))};
    float temp_r {std::max(SIGNf(m_particles.dx3(p)) + temp_i, static_cast<float>(temp_i)) - 1.0f};
    temp_i = static_cast<int>(temp_r);
    m_particles.i3(p) = m_particles.i3(p) + temp_i;
    m_particles.dx3(p) = m_particles.dx3(p) - temp_r;
  }

  // * * * * * * * * * * * * * * *
  // Boris velocity update
  // * * * * * * * * * * * * * * *
  template <Dimension D>
  Inline void Pusher<D>::BorisUpdate(
    const index_t& p, real_t& e0_x1, real_t& e0_x2, real_t& e0_x3, real_t& b0_x1, real_t& b0_x2, real_t& b0_x3) const {
    real_t COEFF {m_coeff};

    e0_x1 *= COEFF;
    e0_x2 *= COEFF;
    e0_x3 *= COEFF;

    real_t u0_x1 {m_particles.ux1(p) + e0_x1};
    real_t u0_x2 {m_particles.ux2(p) + e0_x2};
    real_t u0_x3 {m_particles.ux3(p) + e0_x3};

    COEFF *= 1.0 / std::sqrt(1.0 + u0_x1 * u0_x1 + u0_x2 * u0_x2 + u0_x3 * u0_x3);
    b0_x1 *= COEFF;
    b0_x2 *= COEFF;
    b0_x3 *= COEFF;
    COEFF = 2.0 / (1.0 + b0_x1 * b0_x1 + b0_x2 * b0_x2 + b0_x3 * b0_x3);
    real_t u1_x1 {(u0_x1 + u0_x2 * b0_x3 - u0_x3 * b0_x2) * COEFF};
    real_t u1_x2 {(u0_x2 + u0_x3 * b0_x1 - u0_x1 * b0_x3) * COEFF};
    real_t u1_x3 {(u0_x3 + u0_x1 * b0_x2 - u0_x2 * b0_x1) * COEFF};

    u0_x1 += u1_x2 * b0_x3 - u1_x3 * b0_x2 + e0_x1;
    u0_x2 += u1_x3 * b0_x1 - u1_x1 * b0_x3 + e0_x2;
    u0_x3 += u1_x1 * b0_x2 - u1_x2 * b0_x1 + e0_x3;

    m_particles.ux1(p) = u0_x1;
    m_particles.ux2(p) = u0_x2;
    m_particles.ux3(p) = u0_x3;
  }

  // * * * * * * * * * * * * * * *
  // Field interpolations
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Pusher<Dimension::ONE_D>::interpolateFields(
    const index_t& p, real_t& e0_x1, real_t& e0_x2, real_t& e0_x3, real_t& b0_x1, real_t& b0_x2, real_t& b0_x3) const {
    const auto i {m_particles.i1(p)};
    const real_t dx1 {static_cast<real_t>(m_particles.dx1(p))};

    // first order
    real_t c0, c1;

    // Ex1
    // interpolate to nodes
    c0 = HALF * (m_mblock.em(i, em::ex1) + m_mblock.em(i - 1, em::ex1));
    c1 = HALF * (m_mblock.em(i, em::ex1) + m_mblock.em(i + 1, em::ex1));
    // interpolate from nodes to the particle position
    e0_x1 = c0 * (ONE - dx1) + c1 * dx1;
    // Ex2
    c0 = m_mblock.em(i, em::ex2);
    c1 = m_mblock.em(i + 1, em::ex2);
    e0_x2 = c0 * (ONE - dx1) + c1 * dx1;
    // Ex3
    c0 = m_mblock.em(i, em::ex3);
    c1 = m_mblock.em(i + 1, em::ex3);
    e0_x3 = c0 * (ONE - dx1) + c1 * dx1;

    // Bx1
    c0 = m_mblock.em(i, em::bx1);
    c1 = m_mblock.em(i + 1, em::bx1);
    b0_x1 = c0 * (ONE - dx1) + c1 * dx1;
    // Bx2
    c0 = HALF * (m_mblock.em(i - 1, em::bx2) + m_mblock.em(i, em::bx2));
    c1 = HALF * (m_mblock.em(i, em::bx2) + m_mblock.em(i + 1, em::bx2));
    b0_x2 = c0 * (ONE - dx1) + c1 * dx1;
    // Bx3
    c0 = HALF * (m_mblock.em(i - 1, em::bx3) + m_mblock.em(i, em::bx3));
    c1 = HALF * (m_mblock.em(i, em::bx3) + m_mblock.em(i + 1, em::bx3));
    b0_x3 = c0 * (ONE - dx1) + c1 * dx1;
  }

  template <>
  Inline void Pusher<Dimension::TWO_D>::interpolateFields(
    const index_t& p, real_t& e0_x1, real_t& e0_x2, real_t& e0_x3, real_t& b0_x1, real_t& b0_x2, real_t& b0_x3) const {
    const auto i {m_particles.i1(p)};
    const real_t dx1 {static_cast<real_t>(m_particles.dx1(p))};
    const auto j {m_particles.i2(p)};
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
    c00 = c000 * (ONE - dx1) + c100 * dx1;
    c10 = c010 * (ONE - dx1) + c110 * dx1;
    e0_x1 = c00 * (ONE - dx2) + c10 * dx2;
    // Ex2
    c000 = HALF * (m_mblock.em(i, j, em::ex2) + m_mblock.em(i, j - 1, em::ex2));
    c100 = HALF * (m_mblock.em(i + 1, j, em::ex2) + m_mblock.em(i + 1, j - 1, em::ex2));
    c010 = HALF * (m_mblock.em(i, j, em::ex2) + m_mblock.em(i, j + 1, em::ex2));
    c110 = HALF * (m_mblock.em(i + 1, j, em::ex2) + m_mblock.em(i + 1, j + 1, em::ex2));
    c00 = c000 * (ONE - dx1) + c100 * dx1;
    c10 = c010 * (ONE - dx1) + c110 * dx1;
    e0_x2 = c00 * (ONE - dx2) + c10 * dx2;
    // Ex3
    c000 = m_mblock.em(i, j, em::ex3);
    c100 = m_mblock.em(i + 1, j, em::ex3);
    c010 = m_mblock.em(i, j + 1, em::ex3);
    c110 = m_mblock.em(i + 1, j + 1, em::ex3);
    c00 = c000 * (ONE - dx1) + c100 * dx1;
    c10 = c010 * (ONE - dx1) + c110 * dx1;
    e0_x3 = c00 * (ONE - dx2) + c10 * dx2;

    // Bx1
    c000 = HALF * (m_mblock.em(i, j, em::bx1) + m_mblock.em(i, j - 1, em::bx1));
    c100 = HALF * (m_mblock.em(i + 1, j, em::bx1) + m_mblock.em(i + 1, j - 1, em::bx1));
    c010 = HALF * (m_mblock.em(i, j, em::bx1) + m_mblock.em(i, j + 1, em::bx1));
    c110 = HALF * (m_mblock.em(i + 1, j, em::bx1) + m_mblock.em(i + 1, j + 1, em::bx1));
    c00 = c000 * (ONE - dx1) + c100 * dx1;
    c10 = c010 * (ONE - dx1) + c110 * dx1;
    b0_x1 = c00 * (ONE - dx2) + c10 * dx2;
    // Bx2
    c000 = HALF * (m_mblock.em(i - 1, j, em::bx2) + m_mblock.em(i, j, em::bx2));
    c100 = HALF * (m_mblock.em(i, j, em::bx2) + m_mblock.em(i + 1, j, em::bx2));
    c010 = HALF * (m_mblock.em(i - 1, j + 1, em::bx2) + m_mblock.em(i, j + 1, em::bx2));
    c110 = HALF * (m_mblock.em(i, j + 1, em::bx2) + m_mblock.em(i + 1, j + 1, em::bx2));
    c00 = c000 * (ONE - dx1) + c100 * dx1;
    c10 = c010 * (ONE - dx1) + c110 * dx1;
    b0_x2 = c00 * (ONE - dx2) + c10 * dx2;
    // Bx3
    c000 = QUARTER
           * (m_mblock.em(i - 1, j - 1, em::bx3) + m_mblock.em(i - 1, j, em::bx3) + m_mblock.em(i, j - 1, em::bx3)
              + m_mblock.em(i, j, em::bx3));
    c100 = QUARTER
           * (m_mblock.em(i, j - 1, em::bx3) + m_mblock.em(i, j, em::bx3) + m_mblock.em(i + 1, j - 1, em::bx3)
              + m_mblock.em(i + 1, j, em::bx3));
    c010 = QUARTER
           * (m_mblock.em(i - 1, j, em::bx3) + m_mblock.em(i - 1, j + 1, em::bx3) + m_mblock.em(i, j, em::bx3)
              + m_mblock.em(i, j + 1, em::bx3));
    c110 = QUARTER
           * (m_mblock.em(i, j, em::bx3) + m_mblock.em(i, j + 1, em::bx3) + m_mblock.em(i + 1, j, em::bx3)
              + m_mblock.em(i + 1, j + 1, em::bx3));
    c00 = c000 * (ONE - dx1) + c100 * dx1;
    c10 = c010 * (ONE - dx1) + c110 * dx1;
    b0_x3 = c00 * (ONE - dx2) + c10 * dx2;
  }

  template <>
  Inline void Pusher<Dimension::THREE_D>::interpolateFields(
    const index_t& p, real_t& e0_x1, real_t& e0_x2, real_t& e0_x3, real_t& b0_x1, real_t& b0_x2, real_t& b0_x3) const {
    const auto i {m_particles.i1(p)};
    const real_t dx1 {static_cast<real_t>(m_particles.dx1(p))};
    const auto j {m_particles.i2(p)};
    const real_t dx2 {static_cast<real_t>(m_particles.dx2(p))};
    const auto k {m_particles.i3(p)};
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
    c0 = c00 * (ONE - dx2) + c10 * dx2;
    // interpolate to nodes
    c001 = HALF * (m_mblock.em(i, j, k + 1, em::ex1) + m_mblock.em(i - 1, j, k + 1, em::ex1));
    c101 = HALF * (m_mblock.em(i, j, k + 1, em::ex1) + m_mblock.em(i + 1, j, k + 1, em::ex1));
    c011 = HALF * (m_mblock.em(i, j + 1, k + 1, em::ex1) + m_mblock.em(i - 1, j + 1, k + 1, em::ex1));
    c111 = HALF * (m_mblock.em(i, j + 1, k + 1, em::ex1) + m_mblock.em(i + 1, j + 1, k + 1, em::ex1));
    // interpolate from nodes to the particle position
    c01 = c001 * (ONE - dx1) + c101 * dx1;
    c11 = c011 * (ONE - dx1) + c111 * dx1;
    c1 = c01 * (ONE - dx2) + c11 * dx2;
    e0_x1 = c0 * (ONE - dx3) + c1 * dx3;

    // Ex2
    c000 = HALF * (m_mblock.em(i, j, k, em::ex2) + m_mblock.em(i, j - 1, k, em::ex2));
    c100 = HALF * (m_mblock.em(i + 1, j, k, em::ex2) + m_mblock.em(i + 1, j - 1, k, em::ex2));
    c010 = HALF * (m_mblock.em(i, j, k, em::ex2) + m_mblock.em(i, j + 1, k, em::ex2));
    c110 = HALF * (m_mblock.em(i + 1, j, k, em::ex2) + m_mblock.em(i + 1, j + 1, k, em::ex2));
    c00 = c000 * (ONE - dx1) + c100 * dx1;
    c10 = c010 * (ONE - dx1) + c110 * dx1;
    c0 = c00 * (ONE - dx2) + c10 * dx2;
    c001 = HALF * (m_mblock.em(i, j, k + 1, em::ex2) + m_mblock.em(i, j - 1, k + 1, em::ex2));
    c101 = HALF * (m_mblock.em(i + 1, j, k + 1, em::ex2) + m_mblock.em(i + 1, j - 1, k + 1, em::ex2));
    c011 = HALF * (m_mblock.em(i, j, k + 1, em::ex2) + m_mblock.em(i, j + 1, k + 1, em::ex2));
    c111 = HALF * (m_mblock.em(i + 1, j, k + 1, em::ex2) + m_mblock.em(i + 1, j + 1, k + 1, em::ex2));
    c01 = c001 * (ONE - dx1) + c101 * dx1;
    c11 = c011 * (ONE - dx1) + c111 * dx1;
    c1 = c01 * (ONE - dx2) + c11 * dx2;
    e0_x2 = c0 * (ONE - dx3) + c1 * dx3;

    // Ex3
    c000 = HALF * (m_mblock.em(i, j, k, em::ex3) + m_mblock.em(i, j, k - 1, em::ex3));
    c100 = HALF * (m_mblock.em(i + 1, j, k, em::ex3) + m_mblock.em(i + 1, j, k - 1, em::ex3));
    c010 = HALF * (m_mblock.em(i, j + 1, k, em::ex3) + m_mblock.em(i, j + 1, k - 1, em::ex3));
    c110 = HALF * (m_mblock.em(i + 1, j + 1, k, em::ex3) + m_mblock.em(i + 1, j + 1, k - 1, em::ex3));
    c001 = HALF * (m_mblock.em(i, j, k, em::ex3) + m_mblock.em(i, j, k + 1, em::ex3));
    c101 = HALF * (m_mblock.em(i + 1, j, k, em::ex3) + m_mblock.em(i + 1, j, k + 1, em::ex3));
    c011 = HALF * (m_mblock.em(i, j + 1, k, em::ex3) + m_mblock.em(i, j + 1, k + 1, em::ex3));
    c111 = HALF * (m_mblock.em(i + 1, j + 1, k, em::ex3) + m_mblock.em(i + 1, j + 1, k + 1, em::ex3));
    c00 = c000 * (ONE - dx1) + c100 * dx1;
    c01 = c001 * (ONE - dx1) + c101 * dx1;
    c10 = c010 * (ONE - dx1) + c110 * dx1;
    c11 = c011 * (ONE - dx1) + c111 * dx1;
    c0 = c00 * (ONE - dx2) + c10 * dx2;
    c1 = c01 * (ONE - dx2) + c11 * dx2;
    e0_x3 = c0 * (ONE - dx3) + c1 * dx3;

    // Bx1
    c000 = QUARTER
           * (m_mblock.em(i, j, k, em::bx1) + m_mblock.em(i, j - 1, k, em::bx1) + m_mblock.em(i, j, k - 1, em::bx1)
              + m_mblock.em(i, j - 1, k - 1, em::bx1));
    c100 = QUARTER
           * (m_mblock.em(i + 1, j, k, em::bx1) + m_mblock.em(i + 1, j - 1, k, em::bx1)
              + m_mblock.em(i + 1, j, k - 1, em::bx1) + m_mblock.em(i + 1, j - 1, k - 1, em::bx1));
    c001 = QUARTER
           * (m_mblock.em(i, j, k, em::bx1) + m_mblock.em(i, j, k + 1, em::bx1) + m_mblock.em(i, j - 1, k, em::bx1)
              + m_mblock.em(i, j - 1, k + 1, em::bx1));
    c101 = QUARTER
           * (m_mblock.em(i + 1, j, k, em::bx1) + m_mblock.em(i + 1, j, k + 1, em::bx1)
              + m_mblock.em(i + 1, j - 1, k, em::bx1) + m_mblock.em(i + 1, j - 1, k + 1, em::bx1));
    c010 = QUARTER
           * (m_mblock.em(i, j, k, em::bx1) + m_mblock.em(i, j + 1, k, em::bx1) + m_mblock.em(i, j, k - 1, em::bx1)
              + m_mblock.em(i, j + 1, k - 1, em::bx1));
    c110 = QUARTER
           * (m_mblock.em(i + 1, j, k, em::bx1) + m_mblock.em(i + 1, j, k - 1, em::bx1)
              + m_mblock.em(i + 1, j + 1, k - 1, em::bx1) + m_mblock.em(i + 1, j + 1, k, em::bx1));
    c011 = QUARTER
           * (m_mblock.em(i, j, k, em::bx1) + m_mblock.em(i, j + 1, k, em::bx1) + m_mblock.em(i, j + 1, k + 1, em::bx1)
              + m_mblock.em(i, j, k + 1, em::bx1));
    c111 = QUARTER
           * (m_mblock.em(i + 1, j, k, em::bx1) + m_mblock.em(i + 1, j + 1, k, em::bx1)
              + m_mblock.em(i + 1, j + 1, k + 1, em::bx1) + m_mblock.em(i + 1, j, k + 1, em::bx1));
    c00 = c000 * (ONE - dx1) + c100 * dx1;
    c01 = c001 * (ONE - dx1) + c101 * dx1;
    c10 = c010 * (ONE - dx1) + c110 * dx1;
    c11 = c011 * (ONE - dx1) + c111 * dx1;
    c0 = c00 * (ONE - dx2) + c10 * dx2;
    c1 = c01 * (ONE - dx2) + c11 * dx2;
    b0_x1 = c0 * (ONE - dx3) + c1 * dx3;

    // Bx2
    c000 = QUARTER
           * (m_mblock.em(i - 1, j, k - 1, em::bx2) + m_mblock.em(i - 1, j, k, em::bx2)
              + m_mblock.em(i, j, k - 1, em::bx2) + m_mblock.em(i, j, k, em::bx2));
    c100 = QUARTER
           * (m_mblock.em(i, j, k - 1, em::bx2) + m_mblock.em(i, j, k, em::bx2) + m_mblock.em(i + 1, j, k - 1, em::bx2)
              + m_mblock.em(i + 1, j, k, em::bx2));
    c001 = QUARTER
           * (m_mblock.em(i - 1, j, k, em::bx2) + m_mblock.em(i - 1, j, k + 1, em::bx2) + m_mblock.em(i, j, k, em::bx2)
              + m_mblock.em(i, j, k + 1, em::bx2));
    c101 = QUARTER
           * (m_mblock.em(i, j, k, em::bx2) + m_mblock.em(i, j, k + 1, em::bx2) + m_mblock.em(i + 1, j, k, em::bx2)
              + m_mblock.em(i + 1, j, k + 1, em::bx2));
    c010 = QUARTER
           * (m_mblock.em(i - 1, j + 1, k - 1, em::bx2) + m_mblock.em(i - 1, j + 1, k, em::bx2)
              + m_mblock.em(i, j + 1, k - 1, em::bx2) + m_mblock.em(i, j + 1, k, em::bx2));
    c110 = QUARTER
           * (m_mblock.em(i, j + 1, k - 1, em::bx2) + m_mblock.em(i, j + 1, k, em::bx2)
              + m_mblock.em(i + 1, j + 1, k - 1, em::bx2) + m_mblock.em(i + 1, j + 1, k, em::bx2));
    c011 = QUARTER
           * (m_mblock.em(i - 1, j + 1, k, em::bx2) + m_mblock.em(i - 1, j + 1, k + 1, em::bx2)
              + m_mblock.em(i, j + 1, k, em::bx2) + m_mblock.em(i, j + 1, k + 1, em::bx2));
    c111 = QUARTER
           * (m_mblock.em(i, j + 1, k, em::bx2) + m_mblock.em(i, j + 1, k + 1, em::bx2)
              + m_mblock.em(i + 1, j + 1, k, em::bx2) + m_mblock.em(i + 1, j + 1, k + 1, em::bx2));
    c00 = c000 * (ONE - dx1) + c100 * dx1;
    c01 = c001 * (ONE - dx1) + c101 * dx1;
    c10 = c010 * (ONE - dx1) + c110 * dx1;
    c11 = c011 * (ONE - dx1) + c111 * dx1;
    c0 = c00 * (ONE - dx2) + c10 * dx2;
    c1 = c01 * (ONE - dx2) + c11 * dx2;
    b0_x2 = c0 * (ONE - dx3) + c1 * dx3;

    // Bx3
    c000 = QUARTER
           * (m_mblock.em(i - 1, j - 1, k, em::bx3) + m_mblock.em(i - 1, j, k, em::bx3)
              + m_mblock.em(i, j - 1, k, em::bx3) + m_mblock.em(i, j, k, em::bx3));
    c100 = QUARTER
           * (m_mblock.em(i, j - 1, k, em::bx3) + m_mblock.em(i, j, k, em::bx3) + m_mblock.em(i + 1, j - 1, k, em::bx3)
              + m_mblock.em(i + 1, j, k, em::bx3));
    c001 = QUARTER
           * (m_mblock.em(i - 1, j - 1, k + 1, em::bx3) + m_mblock.em(i - 1, j, k + 1, em::bx3)
              + m_mblock.em(i, j - 1, k + 1, em::bx3) + m_mblock.em(i, j, k + 1, em::bx3));
    c101 = QUARTER
           * (m_mblock.em(i, j - 1, k + 1, em::bx3) + m_mblock.em(i, j, k + 1, em::bx3)
              + m_mblock.em(i + 1, j - 1, k + 1, em::bx3) + m_mblock.em(i + 1, j, k + 1, em::bx3));
    c010 = QUARTER
           * (m_mblock.em(i - 1, j, k, em::bx3) + m_mblock.em(i - 1, j + 1, k, em::bx3) + m_mblock.em(i, j, k, em::bx3)
              + m_mblock.em(i, j + 1, k, em::bx3));
    c110 = QUARTER
           * (m_mblock.em(i, j, k, em::bx3) + m_mblock.em(i, j + 1, k, em::bx3) + m_mblock.em(i + 1, j, k, em::bx3)
              + m_mblock.em(i + 1, j + 1, k, em::bx3));
    c011 = QUARTER
           * (m_mblock.em(i - 1, j, k + 1, em::bx3) + m_mblock.em(i - 1, j + 1, k + 1, em::bx3)
              + m_mblock.em(i, j, k + 1, em::bx3) + m_mblock.em(i, j + 1, k + 1, em::bx3));
    c111 = QUARTER
           * (m_mblock.em(i, j, k + 1, em::bx3) + m_mblock.em(i, j + 1, k + 1, em::bx3)
              + m_mblock.em(i + 1, j, k + 1, em::bx3) + m_mblock.em(i + 1, j + 1, k + 1, em::bx3));
    c00 = c000 * (ONE - dx1) + c100 * dx1;
    c01 = c001 * (ONE - dx1) + c101 * dx1;
    c10 = c010 * (ONE - dx1) + c110 * dx1;
    c11 = c011 * (ONE - dx1) + c111 * dx1;
    c0 = c00 * (ONE - dx2) + c10 * dx2;
    c1 = c01 * (ONE - dx2) + c11 * dx2;
    b0_x3 = c0 * (ONE - dx3) + c1 * dx3;
  }

} // namespace ntt

#endif