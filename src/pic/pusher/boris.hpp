#ifndef PIC_PUSHER_BORIS_H
#define PIC_PUSHER_BORIS_H

#include "global.h"
#include "meshblock.h"
#include "pusher.h"

#include <cmath>

namespace ntt {

template <Dimension D>
class Boris : public Pusher<D> {
  using index_t = NTTArray<real_t*>::size_type;

public:
  Boris(
      const Meshblock<D>& m_meshblock_,
      const Particles<D>& m_particles_,
      const real_t& coeff_,
      const real_t& dt_)
      : Pusher<D> {m_meshblock_, m_particles_, coeff_, dt_} {}

  Inline void operator()(const index_t p) const;

  void velocityUpdate(
      const index_t& p,
      real_t& e0_x1,
      real_t& e0_x2,
      real_t& e0_x3,
      real_t& b0_x1,
      real_t& b0_x2,
      real_t& b0_x3) const {
    real_t COEFF {this->coeff};

    e0_x1 *= COEFF;
    e0_x2 *= COEFF;
    e0_x3 *= COEFF;

    real_t u0_x1 {this->m_particles.m_ux1(p) + e0_x1};
    real_t u0_x2 {this->m_particles.m_ux2(p) + e0_x2};
    real_t u0_x3 {this->m_particles.m_ux3(p) + e0_x3};

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

    this->m_particles.m_ux1(p) = u0_x1;
    this->m_particles.m_ux2(p) = u0_x2;
    this->m_particles.m_ux3(p) = u0_x3;
  }
};

template <>
Inline void Boris<ONE_D>::operator()(const index_t p) const {
  // TESTPERF: this routine vs explicit call
  auto [i, dx1] = convert_x1TOidx1(m_meshblock, m_particles.m_x1(p));
  // dummy fields (no interp)
  real_t e0_x1 {m_meshblock.ex1(i)}, e0_x2 {m_meshblock.ex2(i)}, e0_x3 {m_meshblock.ex3(i)};
  real_t b0_x1 {m_meshblock.bx1(i)}, b0_x2 {m_meshblock.bx2(i)}, b0_x3 {m_meshblock.bx3(i)};
  // TESTPERF: this vs explicit call
  velocityUpdate(p, e0_x1, e0_x2, e0_x3, b0_x1, b0_x2, b0_x3);
  // TESTPERF: faster sqrt?
  real_t inv_gamma0 {
      static_cast<real_t>(1.0)
      / std::sqrt(
          static_cast<real_t>(1.0) + m_particles.m_ux1(p) * m_particles.m_ux1(p)
          + m_particles.m_ux2(p) * m_particles.m_ux2(p)
          + m_particles.m_ux3(p) * m_particles.m_ux3(p))};
  m_particles.m_x1(p) += dt * m_particles.m_ux1(p) * inv_gamma0;
}

template <>
Inline void Boris<TWO_D>::operator()(const index_t p) const {
  auto [i, dx1] = convert_x1TOidx1(m_meshblock, m_particles.m_x1(p));
  auto [j, dx2] = convert_x2TOjdx2(m_meshblock, m_particles.m_x2(p));
  // dummy fields (no interp)
  real_t e0_x1 {m_meshblock.ex1(i, j)}, e0_x2 {m_meshblock.ex2(i, j)},
      e0_x3 {m_meshblock.ex3(i, j)};
  real_t b0_x1 {m_meshblock.bx1(i, j)}, b0_x2 {m_meshblock.bx2(i, j)},
      b0_x3 {m_meshblock.bx3(i, j)};
  velocityUpdate(p, e0_x1, e0_x2, e0_x3, b0_x1, b0_x2, b0_x3);
  real_t inv_gamma0 {
      static_cast<real_t>(1.0)
      / std::sqrt(
          static_cast<real_t>(1.0) + m_particles.m_ux1(p) * m_particles.m_ux1(p)
          + m_particles.m_ux2(p) * m_particles.m_ux2(p)
          + m_particles.m_ux3(p) * m_particles.m_ux3(p))};
  m_particles.m_x1(p) += dt * m_particles.m_ux1(p) * inv_gamma0;
  m_particles.m_x2(p) += dt * m_particles.m_ux2(p) * inv_gamma0;
}

template <>
Inline void Boris<THREE_D>::operator()(const index_t p) const {
  auto [i, dx1] = convert_x1TOidx1(m_meshblock, m_particles.m_x1(p));
  auto [j, dx2] = convert_x2TOjdx2(m_meshblock, m_particles.m_x2(p));
  auto [k, dx3] = convert_x3TOkdx3(m_meshblock, m_particles.m_x3(p));
  // dummy fields (no interp)
  real_t e0_x1 {m_meshblock.ex1(i, j, k)}, e0_x2 {m_meshblock.ex2(i, j, k)},
      e0_x3 {m_meshblock.ex3(i, j, k)};
  real_t b0_x1 {m_meshblock.bx1(i, j, k)}, b0_x2 {m_meshblock.bx2(i, j, k)},
      b0_x3 {m_meshblock.bx3(i, j, k)};
  velocityUpdate(p, e0_x1, e0_x2, e0_x3, b0_x1, b0_x2, b0_x3);
  real_t inv_gamma0 {
      static_cast<real_t>(1.0)
      / std::sqrt(
          static_cast<real_t>(1.0) + m_particles.m_ux1(p) * m_particles.m_ux1(p)
          + m_particles.m_ux2(p) * m_particles.m_ux2(p)
          + m_particles.m_ux3(p) * m_particles.m_ux3(p))};
  m_particles.m_x1(p) += dt * m_particles.m_ux1(p) * inv_gamma0;
  m_particles.m_x2(p) += dt * m_particles.m_ux2(p) * inv_gamma0;
  m_particles.m_x3(p) += dt * m_particles.m_ux3(p) * inv_gamma0;
}

} // namespace ntt

template class ntt::Boris<ntt::ONE_D>;
template class ntt::Boris<ntt::TWO_D>;
template class ntt::Boris<ntt::THREE_D>;

#endif
