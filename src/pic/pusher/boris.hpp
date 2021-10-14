#ifndef PIC_PUSHER_BORIS_H
#define PIC_PUSHER_BORIS_H

#include "global.h"
#include "meshblock.h"
#include "pusher.h"

#include <cmath>

namespace ntt {

// this depends on coord system

template <Dimension D>
class Boris : public Pusher<D> {
  using index_t = typename Pusher<D>::index_t;
public:
  Boris(
      const Meshblock<D>& m_meshblock_,
      const Particles<D>& m_particles_,
      const real_t& coeff_,
      const real_t& dt_)
      : Pusher<D> {m_meshblock_, m_particles_, coeff_, dt_} {}

  void velocityUpdate(
      const index_t& p,
      real_t& e0_x1,
      real_t& e0_x2,
      real_t& e0_x3,
      real_t& b0_x1,
      real_t& b0_x2,
      real_t& b0_x3) const override {
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

} // namespace ntt

template class ntt::Boris<ntt::ONE_D>;
template class ntt::Boris<ntt::TWO_D>;
template class ntt::Boris<ntt::THREE_D>;

#endif
