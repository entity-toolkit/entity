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

  // clang-format off
  void interpolateFields(const index_t&,
                         real_t&, real_t&, real_t&,
                         real_t&, real_t&, real_t&) const;
  void positionUpdate(const index_t&) const;
  virtual void velocityUpdate(const index_t&,
                              real_t&, real_t&, real_t&,
                              real_t&, real_t&, real_t&) const {}
  Inline void operator()(const index_t p) const {
    real_t e0_x1, e0_x2, e0_x3;
    real_t b0_x1, b0_x2, b0_x3;
    interpolateFields(p,
                      e0_x1, e0_x2, e0_x3,
                      b0_x1, b0_x2, b0_x3);
    velocityUpdate(p,
                   e0_x1, e0_x2, e0_x3,
                   b0_x1, b0_x2, b0_x3);
    positionUpdate(p);
  }
  // clang-format on
};

// * * * * Position update * * * * * * * * * * * * * *
template <>
void Pusher<ONE_D>::positionUpdate(const index_t& p) const {
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
void Pusher<TWO_D>::positionUpdate(const index_t& p) const {
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
void Pusher<THREE_D>::positionUpdate(const index_t& p) const {
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
  const auto [i, dx1] = convert_x1TOidx1(m_meshblock, m_particles.m_x1(p));
  e0_x1 = m_meshblock.ex1(i);
  e0_x2 = m_meshblock.ex2(i);
  e0_x3 = m_meshblock.ex3(i);

  b0_x1 = m_meshblock.bx1(i);
  b0_x2 = m_meshblock.bx2(i);
  b0_x3 = m_meshblock.bx3(i);
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
  const auto [i, dx1] = convert_x1TOidx1(m_meshblock, m_particles.m_x1(p));
  const auto [j, dx2] = convert_x2TOjdx2(m_meshblock, m_particles.m_x2(p));
  e0_x1 = m_meshblock.ex1(i, j);
  e0_x2 = m_meshblock.ex2(i, j);
  e0_x3 = m_meshblock.ex3(i, j);

  b0_x1 = m_meshblock.bx1(i, j);
  b0_x2 = m_meshblock.bx2(i, j);
  b0_x3 = m_meshblock.bx3(i, j);
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
  const auto [i, dx1] = convert_x1TOidx1(m_meshblock, m_particles.m_x1(p));
  const auto [j, dx2] = convert_x2TOjdx2(m_meshblock, m_particles.m_x2(p));
  const auto [k, dx3] = convert_x3TOkdx3(m_meshblock, m_particles.m_x3(p));
  e0_x1 = m_meshblock.ex1(i, j, k);
  e0_x2 = m_meshblock.ex2(i, j, k);
  e0_x3 = m_meshblock.ex3(i, j, k);

  b0_x1 = m_meshblock.bx1(i, j, k);
  b0_x2 = m_meshblock.bx2(i, j, k);
  b0_x3 = m_meshblock.bx3(i, j, k);
}

} // namespace ntt

template class ntt::Pusher<ntt::ONE_D>;
template class ntt::Pusher<ntt::TWO_D>;
template class ntt::Pusher<ntt::THREE_D>;

#endif
