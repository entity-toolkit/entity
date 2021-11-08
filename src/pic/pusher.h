#ifndef PIC_PUSHER_H
#define PIC_PUSHER_H

#include "global.h"
#include "meshblock.h"
#include "particles.h"

#include <utility>

namespace ntt {

template <Dimension D>
class Pusher {
public:
  struct Boris_Cartesian_t {};
  struct Boris_Curvilinear_t {};

  struct Photon_Cartesian_t {};
  struct Photon_Curvilinear_t {};

  Meshblock<D> m_meshblock;
  Particles<D> m_particles;
  real_t coeff;
  real_t dt;
  using index_t = NTTArray<real_t*>::size_type;

  Pusher(
      const Meshblock<D>& m_meshblock_,
      const Particles<D>& m_particles_,
      const real_t& coeff_,
      const real_t& dt_)
      : m_meshblock(m_meshblock_), m_particles(m_particles_), coeff(coeff_), dt(dt_) {}

  void pushAllParticles() {
    // TODO: call different options
    if (m_particles.get_mass() == 0) {
      if (m_meshblock.m_coord_system == CARTESIAN_COORD) {
        auto range_policy
            = Kokkos::RangePolicy<AccelExeSpace, Photon_Cartesian_t>(0, m_particles.get_npart());
        Kokkos::parallel_for("pusher", range_policy, *this);
      } else {
        throw std::runtime_error("# Error: Curvilinear pusher not implemented.");
        // auto range_policy
        //     = Kokkos::RangePolicy<AccelExeSpace, Photon_Curvilinear_t>(0,
        //     m_particles.get_npart());
        // Kokkos::parallel_for("pusher", range_policy, *this);
      }
    } else if (m_particles.get_mass() != 0) {
      if (m_meshblock.m_coord_system == CARTESIAN_COORD) {
        auto range_policy
            = Kokkos::RangePolicy<AccelExeSpace, Boris_Cartesian_t>(0, m_particles.get_npart());
        Kokkos::parallel_for("pusher", range_policy, *this);
      } else {
        throw std::runtime_error("# Error: Curvilinear pusher not implemented.");
        // auto range_policy
        //     = Kokkos::RangePolicy<AccelExeSpace, Boris_Curvilinear_t>(0,
        //     m_particles.get_npart());
        // Kokkos::parallel_for("pusher", range_policy, *this);
      }
    }
  }

  // clang-format off
  // * * * common operations * * *
  void interpolateFields(const index_t&,
                         real_t&, real_t&, real_t&,
                         real_t&, real_t&, real_t&) const;
  void positionUpdate(const index_t&) const;

  Inline void operator()(const Boris_Cartesian_t&, const index_t p) const {
    real_t e0_x1, e0_x2, e0_x3;
    real_t b0_x1, b0_x2, b0_x3;
    // TODO:
    // fix this to cartesian
    // 1. interpolate fields
    // 2. convert E,B and X,U to cartesian
    // 3. update U
    // 4. update X
    // 5. convert back
    interpolateFields(p,
                      e0_x1, e0_x2, e0_x3,
                      b0_x1, b0_x2, b0_x3);
    BorisUpdate(p,
                e0_x1, e0_x2, e0_x3,
                b0_x1, b0_x2, b0_x3);
    positionUpdate(p);
  }

  Inline void operator()(const Photon_Cartesian_t&, const index_t p) const {
    positionUpdate(p);
  }

  // velocity updaters
  void BorisUpdate(const index_t&,
                   real_t&, real_t&, real_t&,
                   real_t&, real_t&, real_t&) const;
  // clang-format on
};

// * * * * Position update * * * * * * * * * * * * * *
template <>
Inline void Pusher<ONE_D>::positionUpdate(const index_t& p) const {
  // TESTPERF: faster sqrt?
  // clang-format off
  real_t inv_gamma0 {
      ONE / std::sqrt(ONE
          + m_particles.m_ux1(p) * m_particles.m_ux1(p)
          + m_particles.m_ux2(p) * m_particles.m_ux2(p)
          + m_particles.m_ux3(p) * m_particles.m_ux3(p))
        };
  // clang-format on
  m_particles.m_x1(p) += dt * m_particles.m_ux1(p) * inv_gamma0;
}

template <>
Inline void Pusher<TWO_D>::positionUpdate(const index_t& p) const {
  // clang-format off
  real_t inv_gamma0 {
      ONE / std::sqrt(ONE
          + m_particles.m_ux1(p) * m_particles.m_ux1(p)
          + m_particles.m_ux2(p) * m_particles.m_ux2(p)
          + m_particles.m_ux3(p) * m_particles.m_ux3(p))
        };
  // clang-format on
  m_particles.m_x1(p) += dt * m_particles.m_ux1(p) * inv_gamma0;
  m_particles.m_x2(p) += dt * m_particles.m_ux2(p) * inv_gamma0;
}

template <>
Inline void Pusher<THREE_D>::positionUpdate(const index_t& p) const {
  // clang-format off
  real_t inv_gamma0 {
      ONE / std::sqrt(ONE
          + m_particles.m_ux1(p) * m_particles.m_ux1(p)
          + m_particles.m_ux2(p) * m_particles.m_ux2(p)
          + m_particles.m_ux3(p) * m_particles.m_ux3(p))
        };
  // clang-format on
  m_particles.m_x1(p) += dt * m_particles.m_ux1(p) * inv_gamma0;
  m_particles.m_x2(p) += dt * m_particles.m_ux2(p) * inv_gamma0;
  m_particles.m_x3(p) += dt * m_particles.m_ux3(p) * inv_gamma0;
}

// * * * * Field interpolation * * * * * * * * * * * *
template <>
Inline void Pusher<ONE_D>::interpolateFields(
    const index_t& p,
    real_t& e0_x1,
    real_t& e0_x2,
    real_t& e0_x3,
    real_t& b0_x1,
    real_t& b0_x2,
    real_t& b0_x3) const {
  const auto [i, dx1] = m_meshblock.convert_x1TOidx1(m_particles.m_x1(p));

  // first order
  real_t c0, c1;

  // Ex1
  // interpolate to nodes
  c0 = 0.5 * (m_meshblock.em_fields(i, fld::ex1) + m_meshblock.em_fields(i - 1, fld::ex1));
  c1 = 0.5 * (m_meshblock.em_fields(i, fld::ex1) + m_meshblock.em_fields(i + 1, fld::ex1));
  // interpolate from nodes to the particle position
  e0_x1 = c0 * (ONE - dx1) + c1 * dx1;
  // Ex2
  c0 = m_meshblock.em_fields(i, fld::ex2);
  c1 = m_meshblock.em_fields(i + 1, fld::ex2);
  e0_x2 = c0 * (ONE - dx1) + c1 * dx1;
  // Ex3
  c0 = m_meshblock.em_fields(i, fld::ex3);
  c1 = m_meshblock.em_fields(i + 1, fld::ex3);
  e0_x3 = c0 * (ONE - dx1) + c1 * dx1;

  // Bx1
  c0 = m_meshblock.em_fields(i, fld::bx1);
  c1 = m_meshblock.em_fields(i + 1, fld::bx1);
  b0_x1 = c0 * (ONE - dx1) + c1 * dx1;
  // Bx2
  c0 = 0.5 * (m_meshblock.em_fields(i - 1, fld::bx2) + m_meshblock.em_fields(i, fld::bx2));
  c1 = 0.5 * (m_meshblock.em_fields(i, fld::bx2) + m_meshblock.em_fields(i + 1, fld::bx2));
  b0_x2 = c0 * (ONE - dx1) + c1 * dx1;
  // Bx3
  c0 = 0.5 * (m_meshblock.em_fields(i - 1, fld::bx3) + m_meshblock.em_fields(i, fld::bx3));
  c1 = 0.5 * (m_meshblock.em_fields(i, fld::bx3) + m_meshblock.em_fields(i + 1, fld::bx3));
  b0_x3 = c0 * (ONE - dx1) + c1 * dx1;
}

template <>
Inline void Pusher<TWO_D>::interpolateFields(
    const index_t& p,
    real_t& e0_x1,
    real_t& e0_x2,
    real_t& e0_x3,
    real_t& b0_x1,
    real_t& b0_x2,
    real_t& b0_x3) const {
  // dx1, dx2 are normalized to cell sizes
  const auto [i, dx1] = m_meshblock.convert_x1TOidx1(m_particles.m_x1(p));
  const auto [j, dx2] = m_meshblock.convert_x2TOjdx2(m_particles.m_x2(p));

  // first order
  real_t c000, c100, c010, c110, c00, c10;

  // clang-format off
  // Ex1
  // interpolate to nodes
  c000 = 0.5 * (m_meshblock.em_fields(i, j, fld::ex1) + m_meshblock.em_fields(i - 1, j, fld::ex1));
  c100 = 0.5 * (m_meshblock.em_fields(i, j, fld::ex1) + m_meshblock.em_fields(i + 1, j, fld::ex1));
  c010 = 0.5 * (m_meshblock.em_fields(i, j + 1, fld::ex1) + m_meshblock.em_fields(i - 1, j + 1, fld::ex1));
  c110 = 0.5 * (m_meshblock.em_fields(i, j + 1, fld::ex1) + m_meshblock.em_fields(i + 1, j + 1, fld::ex1));
  // interpolate from nodes to the particle position
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  e0_x1 = c00 * (ONE - dx2) + c10 * dx2;
  // Ex2
  c000 = 0.5 * (m_meshblock.em_fields(i, j, fld::ex2) + m_meshblock.em_fields(i, j - 1, fld::ex2));
  c100 = 0.5 * (m_meshblock.em_fields(i + 1, j, fld::ex2) + m_meshblock.em_fields(i + 1, j - 1, fld::ex2));
  c010 = 0.5 * (m_meshblock.em_fields(i, j, fld::ex2) + m_meshblock.em_fields(i, j + 1, fld::ex2));
  c110 = 0.5 * (m_meshblock.em_fields(i + 1, j, fld::ex2) + m_meshblock.em_fields(i + 1, j + 1, fld::ex2));
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  e0_x2 = c00 * (ONE - dx2) + c10 * dx2;
  // Ex3
  c000 = m_meshblock.em_fields(i, j, fld::ex3);
  c100 = m_meshblock.em_fields(i + 1, j, fld::ex3);
  c010 = m_meshblock.em_fields(i, j + 1, fld::ex3);
  c110 = m_meshblock.em_fields(i + 1, j + 1, fld::ex3);
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  e0_x3 = c00 * (ONE - dx2) + c10 * dx2;

  // Bx1
  c000 = 0.5 * (m_meshblock.em_fields(i, j, fld::bx1) + m_meshblock.em_fields(i, j - 1, fld::bx1));
  c100 = 0.5 * (m_meshblock.em_fields(i + 1, j, fld::bx1) + m_meshblock.em_fields(i + 1, j - 1, fld::bx1));
  c010 = 0.5 * (m_meshblock.em_fields(i, j, fld::bx1) + m_meshblock.em_fields(i, j + 1, fld::bx1));
  c110 = 0.5 * (m_meshblock.em_fields(i + 1, j, fld::bx1) + m_meshblock.em_fields(i + 1, j + 1, fld::bx1));
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  b0_x1 = c00 * (ONE - dx2) + c10 * dx2;
  // Bx2
  c000 = 0.5 * (m_meshblock.em_fields(i - 1, j, fld::bx2) + m_meshblock.em_fields(i, j, fld::bx2));
  c100 = 0.5 * (m_meshblock.em_fields(i, j, fld::bx2) + m_meshblock.em_fields(i + 1, j, fld::bx2));
  c010 = 0.5 * (m_meshblock.em_fields(i - 1, j + 1, fld::bx2) + m_meshblock.em_fields(i, j + 1, fld::bx2));
  c110 = 0.5 * (m_meshblock.em_fields(i, j + 1, fld::bx2) + m_meshblock.em_fields(i + 1, j + 1, fld::bx2));
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  b0_x2 = c00 * (ONE - dx2) + c10 * dx2;
  // Bx3
  c000 = 0.25
         * (m_meshblock.em_fields(i - 1, j - 1, fld::bx3) + m_meshblock.em_fields(i - 1, j, fld::bx3) + m_meshblock.em_fields(i, j - 1, fld::bx3)
            + m_meshblock.em_fields(i, j, fld::bx3));
  c100 = 0.25
         * (m_meshblock.em_fields(i, j - 1, fld::bx3) + m_meshblock.em_fields(i, j, fld::bx3) + m_meshblock.em_fields(i + 1, j - 1, fld::bx3)
            + m_meshblock.em_fields(i + 1, j, fld::bx3));
  c010 = 0.25
         * (m_meshblock.em_fields(i - 1, j, fld::bx3) + m_meshblock.em_fields(i - 1, j + 1, fld::bx3) + m_meshblock.em_fields(i, j, fld::bx3)
            + m_meshblock.em_fields(i, j + 1, fld::bx3));
  c110 = 0.25
         * (m_meshblock.em_fields(i, j, fld::bx3) + m_meshblock.em_fields(i, j + 1, fld::bx3) + m_meshblock.em_fields(i + 1, j, fld::bx3)
            + m_meshblock.em_fields(i + 1, j + 1, fld::bx3));
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  b0_x3 = c00 * (ONE - dx2) + c10 * dx2;
  // clang-format on
}

template <>
Inline void Pusher<THREE_D>::interpolateFields(
    const index_t& p,
    real_t& e0_x1,
    real_t& e0_x2,
    real_t& e0_x3,
    real_t& b0_x1,
    real_t& b0_x2,
    real_t& b0_x3) const {
  const auto [i, dx1] = m_meshblock.convert_x1TOidx1(m_particles.m_x1(p));
  const auto [j, dx2] = m_meshblock.convert_x2TOjdx2(m_particles.m_x2(p));
  const auto [k, dx3] = m_meshblock.convert_x3TOkdx3(m_particles.m_x3(p));

  // first order
  real_t c000, c100, c010, c110, c001, c101, c011, c111, c00, c10, c01, c11, c0, c1;

  // Ex1
  // interpolate to nodes
  c000
      = 0.5
        * (m_meshblock.em_fields(i, j, k, fld::ex1) + m_meshblock.em_fields(i - 1, j, k, fld::ex1));
  c100
      = 0.5
        * (m_meshblock.em_fields(i, j, k, fld::ex1) + m_meshblock.em_fields(i + 1, j, k, fld::ex1));
  c010 = 0.5
         * (m_meshblock.em_fields(i, j + 1, k, fld::ex1)
            + m_meshblock.em_fields(i - 1, j + 1, k, fld::ex1));
  c110 = 0.5
         * (m_meshblock.em_fields(i, j + 1, k, fld::ex1)
            + m_meshblock.em_fields(i + 1, j + 1, k, fld::ex1));
  // interpolate from nodes to the particle position
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  c0 = c00 * (ONE - dx2) + c10 * dx2;
  // interpolate to nodes
  c001 = 0.5
         * (m_meshblock.em_fields(i, j, k + 1, fld::ex1)
            + m_meshblock.em_fields(i - 1, j, k + 1, fld::ex1));
  c101 = 0.5
         * (m_meshblock.em_fields(i, j, k + 1, fld::ex1)
            + m_meshblock.em_fields(i + 1, j, k + 1, fld::ex1));
  c011 = 0.5
         * (m_meshblock.em_fields(i, j + 1, k + 1, fld::ex1)
            + m_meshblock.em_fields(i - 1, j + 1, k + 1, fld::ex1));
  c111 = 0.5
         * (m_meshblock.em_fields(i, j + 1, k + 1, fld::ex1)
            + m_meshblock.em_fields(i + 1, j + 1, k + 1, fld::ex1));
  // interpolate from nodes to the particle position
  c01 = c001 * (ONE - dx1) + c101 * dx1;
  c11 = c011 * (ONE - dx1) + c111 * dx1;
  c1 = c01 * (ONE - dx2) + c11 * dx2;
  e0_x1 = c0 * (ONE - dx3) + c1 * dx3;

  // Ex2
  c000
      = 0.5
        * (m_meshblock.em_fields(i, j, k, fld::ex2) + m_meshblock.em_fields(i, j - 1, k, fld::ex2));
  c100 = 0.5
         * (m_meshblock.em_fields(i + 1, j, k, fld::ex2)
            + m_meshblock.em_fields(i + 1, j - 1, k, fld::ex2));
  c010
      = 0.5
        * (m_meshblock.em_fields(i, j, k, fld::ex2) + m_meshblock.em_fields(i, j + 1, k, fld::ex2));
  c110 = 0.5
         * (m_meshblock.em_fields(i + 1, j, k, fld::ex2)
            + m_meshblock.em_fields(i + 1, j + 1, k, fld::ex2));
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  c0 = c00 * (ONE - dx2) + c10 * dx2;
  c001 = 0.5
         * (m_meshblock.em_fields(i, j, k + 1, fld::ex2)
            + m_meshblock.em_fields(i, j - 1, k + 1, fld::ex2));
  c101 = 0.5
         * (m_meshblock.em_fields(i + 1, j, k + 1, fld::ex2)
            + m_meshblock.em_fields(i + 1, j - 1, k + 1, fld::ex2));
  c011 = 0.5
         * (m_meshblock.em_fields(i, j, k + 1, fld::ex2)
            + m_meshblock.em_fields(i, j + 1, k + 1, fld::ex2));
  c111 = 0.5
         * (m_meshblock.em_fields(i + 1, j, k + 1, fld::ex2)
            + m_meshblock.em_fields(i + 1, j + 1, k + 1, fld::ex2));
  c01 = c001 * (ONE - dx1) + c101 * dx1;
  c11 = c011 * (ONE - dx1) + c111 * dx1;
  c1 = c01 * (ONE - dx2) + c11 * dx2;
  e0_x2 = c0 * (ONE - dx3) + c1 * dx3;

  // Ex3
  c000
      = 0.5
        * (m_meshblock.em_fields(i, j, k, fld::ex3) + m_meshblock.em_fields(i, j, k - 1, fld::ex3));
  c100 = 0.5
         * (m_meshblock.em_fields(i + 1, j, k, fld::ex3)
            + m_meshblock.em_fields(i + 1, j, k - 1, fld::ex3));
  c010 = 0.5
         * (m_meshblock.em_fields(i, j + 1, k, fld::ex3)
            + m_meshblock.em_fields(i, j + 1, k - 1, fld::ex3));
  c110 = 0.5
         * (m_meshblock.em_fields(i + 1, j + 1, k, fld::ex3)
            + m_meshblock.em_fields(i + 1, j + 1, k - 1, fld::ex3));
  c001
      = 0.5
        * (m_meshblock.em_fields(i, j, k, fld::ex3) + m_meshblock.em_fields(i, j, k + 1, fld::ex3));
  c101 = 0.5
         * (m_meshblock.em_fields(i + 1, j, k, fld::ex3)
            + m_meshblock.em_fields(i + 1, j, k + 1, fld::ex3));
  c011 = 0.5
         * (m_meshblock.em_fields(i, j + 1, k, fld::ex3)
            + m_meshblock.em_fields(i, j + 1, k + 1, fld::ex3));
  c111 = 0.5
         * (m_meshblock.em_fields(i + 1, j + 1, k, fld::ex3)
            + m_meshblock.em_fields(i + 1, j + 1, k + 1, fld::ex3));
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c01 = c001 * (ONE - dx1) + c101 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  c11 = c011 * (ONE - dx1) + c111 * dx1;
  c0 = c00 * (ONE - dx2) + c10 * dx2;
  c1 = c01 * (ONE - dx2) + c11 * dx2;
  e0_x3 = c0 * (ONE - dx3) + c1 * dx3;

  // Bx1
  c000 = 0.25
         * (m_meshblock.em_fields(i, j, k, fld::bx1) + m_meshblock.em_fields(i, j - 1, k, fld::bx1)
            + m_meshblock.em_fields(i, j, k - 1, fld::bx1)
            + m_meshblock.em_fields(i, j - 1, k - 1, fld::bx1));
  c100 = 0.25
         * (m_meshblock.em_fields(i + 1, j, k, fld::bx1)
            + m_meshblock.em_fields(i + 1, j - 1, k, fld::bx1)
            + m_meshblock.em_fields(i + 1, j, k - 1, fld::bx1)
            + m_meshblock.em_fields(i + 1, j - 1, k - 1, fld::bx1));
  c001 = 0.25
         * (m_meshblock.em_fields(i, j, k, fld::bx1) + m_meshblock.em_fields(i, j, k + 1, fld::bx1)
            + m_meshblock.em_fields(i, j - 1, k, fld::bx1)
            + m_meshblock.em_fields(i, j - 1, k + 1, fld::bx1));
  c101 = 0.25
         * (m_meshblock.em_fields(i + 1, j, k, fld::bx1)
            + m_meshblock.em_fields(i + 1, j, k + 1, fld::bx1)
            + m_meshblock.em_fields(i + 1, j - 1, k, fld::bx1)
            + m_meshblock.em_fields(i + 1, j - 1, k + 1, fld::bx1));
  c010 = 0.25
         * (m_meshblock.em_fields(i, j, k, fld::bx1) + m_meshblock.em_fields(i, j + 1, k, fld::bx1)
            + m_meshblock.em_fields(i, j, k - 1, fld::bx1)
            + m_meshblock.em_fields(i, j + 1, k - 1, fld::bx1));
  c110 = 0.25
         * (m_meshblock.em_fields(i + 1, j, k, fld::bx1)
            + m_meshblock.em_fields(i + 1, j, k - 1, fld::bx1)
            + m_meshblock.em_fields(i + 1, j + 1, k - 1, fld::bx1)
            + m_meshblock.em_fields(i + 1, j + 1, k, fld::bx1));
  c011 = 0.25
         * (m_meshblock.em_fields(i, j, k, fld::bx1) + m_meshblock.em_fields(i, j + 1, k, fld::bx1)
            + m_meshblock.em_fields(i, j + 1, k + 1, fld::bx1)
            + m_meshblock.em_fields(i, j, k + 1, fld::bx1));
  c111 = 0.25
         * (m_meshblock.em_fields(i + 1, j, k, fld::bx1)
            + m_meshblock.em_fields(i + 1, j + 1, k, fld::bx1)
            + m_meshblock.em_fields(i + 1, j + 1, k + 1, fld::bx1)
            + m_meshblock.em_fields(i + 1, j, k + 1, fld::bx1));
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c01 = c001 * (ONE - dx1) + c101 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  c11 = c011 * (ONE - dx1) + c111 * dx1;
  c0 = c00 * (ONE - dx2) + c10 * dx2;
  c1 = c01 * (ONE - dx2) + c11 * dx2;
  b0_x1 = c0 * (ONE - dx3) + c1 * dx3;

  // Bx2
  c000 = 0.25
         * (m_meshblock.em_fields(i - 1, j, k - 1, fld::bx2)
            + m_meshblock.em_fields(i - 1, j, k, fld::bx2)
            + m_meshblock.em_fields(i, j, k - 1, fld::bx2)
            + m_meshblock.em_fields(i, j, k, fld::bx2));
  c100 = 0.25
         * (m_meshblock.em_fields(i, j, k - 1, fld::bx2) + m_meshblock.em_fields(i, j, k, fld::bx2)
            + m_meshblock.em_fields(i + 1, j, k - 1, fld::bx2)
            + m_meshblock.em_fields(i + 1, j, k, fld::bx2));
  c001 = 0.25
         * (m_meshblock.em_fields(i - 1, j, k, fld::bx2)
            + m_meshblock.em_fields(i - 1, j, k + 1, fld::bx2)
            + m_meshblock.em_fields(i, j, k, fld::bx2)
            + m_meshblock.em_fields(i, j, k + 1, fld::bx2));
  c101 = 0.25
         * (m_meshblock.em_fields(i, j, k, fld::bx2) + m_meshblock.em_fields(i, j, k + 1, fld::bx2)
            + m_meshblock.em_fields(i + 1, j, k, fld::bx2)
            + m_meshblock.em_fields(i + 1, j, k + 1, fld::bx2));
  c010 = 0.25
         * (m_meshblock.em_fields(i - 1, j + 1, k - 1, fld::bx2)
            + m_meshblock.em_fields(i - 1, j + 1, k, fld::bx2)
            + m_meshblock.em_fields(i, j + 1, k - 1, fld::bx2)
            + m_meshblock.em_fields(i, j + 1, k, fld::bx2));
  c110 = 0.25
         * (m_meshblock.em_fields(i, j + 1, k - 1, fld::bx2)
            + m_meshblock.em_fields(i, j + 1, k, fld::bx2)
            + m_meshblock.em_fields(i + 1, j + 1, k - 1, fld::bx2)
            + m_meshblock.em_fields(i + 1, j + 1, k, fld::bx2));
  c011 = 0.25
         * (m_meshblock.em_fields(i - 1, j + 1, k, fld::bx2)
            + m_meshblock.em_fields(i - 1, j + 1, k + 1, fld::bx2)
            + m_meshblock.em_fields(i, j + 1, k, fld::bx2)
            + m_meshblock.em_fields(i, j + 1, k + 1, fld::bx2));
  c111 = 0.25
         * (m_meshblock.em_fields(i, j + 1, k, fld::bx2)
            + m_meshblock.em_fields(i, j + 1, k + 1, fld::bx2)
            + m_meshblock.em_fields(i + 1, j + 1, k, fld::bx2)
            + m_meshblock.em_fields(i + 1, j + 1, k + 1, fld::bx2));
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c01 = c001 * (ONE - dx1) + c101 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  c11 = c011 * (ONE - dx1) + c111 * dx1;
  c0 = c00 * (ONE - dx2) + c10 * dx2;
  c1 = c01 * (ONE - dx2) + c11 * dx2;
  b0_x2 = c0 * (ONE - dx3) + c1 * dx3;

  // Bx3
  c000 = 0.25
         * (m_meshblock.em_fields(i - 1, j - 1, k, fld::bx3)
            + m_meshblock.em_fields(i - 1, j, k, fld::bx3)
            + m_meshblock.em_fields(i, j - 1, k, fld::bx3)
            + m_meshblock.em_fields(i, j, k, fld::bx3));
  c100 = 0.25
         * (m_meshblock.em_fields(i, j - 1, k, fld::bx3) + m_meshblock.em_fields(i, j, k, fld::bx3)
            + m_meshblock.em_fields(i + 1, j - 1, k, fld::bx3)
            + m_meshblock.em_fields(i + 1, j, k, fld::bx3));
  c001 = 0.25
         * (m_meshblock.em_fields(i - 1, j - 1, k + 1, fld::bx3)
            + m_meshblock.em_fields(i - 1, j, k + 1, fld::bx3)
            + m_meshblock.em_fields(i, j - 1, k + 1, fld::bx3)
            + m_meshblock.em_fields(i, j, k + 1, fld::bx3));
  c101 = 0.25
         * (m_meshblock.em_fields(i, j - 1, k + 1, fld::bx3)
            + m_meshblock.em_fields(i, j, k + 1, fld::bx3)
            + m_meshblock.em_fields(i + 1, j - 1, k + 1, fld::bx3)
            + m_meshblock.em_fields(i + 1, j, k + 1, fld::bx3));
  c010 = 0.25
         * (m_meshblock.em_fields(i - 1, j, k, fld::bx3)
            + m_meshblock.em_fields(i - 1, j + 1, k, fld::bx3)
            + m_meshblock.em_fields(i, j, k, fld::bx3)
            + m_meshblock.em_fields(i, j + 1, k, fld::bx3));
  c110 = 0.25
         * (m_meshblock.em_fields(i, j, k, fld::bx3) + m_meshblock.em_fields(i, j + 1, k, fld::bx3)
            + m_meshblock.em_fields(i + 1, j, k, fld::bx3)
            + m_meshblock.em_fields(i + 1, j + 1, k, fld::bx3));
  c011 = 0.25
         * (m_meshblock.em_fields(i - 1, j, k + 1, fld::bx3)
            + m_meshblock.em_fields(i - 1, j + 1, k + 1, fld::bx3)
            + m_meshblock.em_fields(i, j, k + 1, fld::bx3)
            + m_meshblock.em_fields(i, j + 1, k + 1, fld::bx3));
  c111 = 0.25
         * (m_meshblock.em_fields(i, j, k + 1, fld::bx3)
            + m_meshblock.em_fields(i, j + 1, k + 1, fld::bx3)
            + m_meshblock.em_fields(i + 1, j, k + 1, fld::bx3)
            + m_meshblock.em_fields(i + 1, j + 1, k + 1, fld::bx3));
  c00 = c000 * (ONE - dx1) + c100 * dx1;
  c01 = c001 * (ONE - dx1) + c101 * dx1;
  c10 = c010 * (ONE - dx1) + c110 * dx1;
  c11 = c011 * (ONE - dx1) + c111 * dx1;
  c0 = c00 * (ONE - dx2) + c10 * dx2;
  c1 = c01 * (ONE - dx2) + c11 * dx2;
  b0_x3 = c0 * (ONE - dx3) + c1 * dx3;
}

template <Dimension D>
Inline void Pusher<D>::BorisUpdate(
    const index_t& p,
    real_t& e0_x1,
    real_t& e0_x2,
    real_t& e0_x3,
    real_t& b0_x1,
    real_t& b0_x2,
    real_t& b0_x3) const {
  real_t COEFF {coeff};

  e0_x1 *= COEFF;
  e0_x2 *= COEFF;
  e0_x3 *= COEFF;

  real_t u0_x1 {m_particles.m_ux1(p) + e0_x1};
  real_t u0_x2 {m_particles.m_ux2(p) + e0_x2};
  real_t u0_x3 {m_particles.m_ux3(p) + e0_x3};

  // TESTPERF: faster sqrt?
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

  m_particles.m_ux1(p) = u0_x1;
  m_particles.m_ux2(p) = u0_x2;
  m_particles.m_ux3(p) = u0_x3;
}

} // namespace ntt

template class ntt::Pusher<ntt::ONE_D>;
template class ntt::Pusher<ntt::TWO_D>;
template class ntt::Pusher<ntt::THREE_D>;

#endif