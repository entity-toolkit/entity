#ifndef PIC_PUSHER_BORIS_H
#define PIC_PUSHER_BORIS_H

#include "global.h"
#include "meshblock.h"
#include "pusher.h"

#include <cmath>

namespace ntt {

class Boris1D : public Pusher<ONE_D> {
public:
  Boris1D(const Meshblock<ONE_D>& m_meshblock_,
          const Particles<ONE_D>& m_particles_,
          const real_t& coeff_,
          const real_t& dt_)
      : Pusher<ONE_D>{m_meshblock_, m_particles_, coeff_, dt_} {}
  Inline void operator()(const index_t p) const {
    // dummy fields
    real_t e0_x1{0.0}, e0_x2{0.1}, e0_x3{0.0};
    real_t b0_x1{0.0}, b0_x2{0.0}, b0_x3{1.0};
    real_t COEFF{coeff};

    e0_x1 *= COEFF;
    e0_x2 *= COEFF;
    e0_x3 *= COEFF;

    real_t u0_x1{m_particles.m_ux1(p) + e0_x1};
    real_t u0_x2{m_particles.m_ux2(p) + e0_x2};
    real_t u0_x3{m_particles.m_ux3(p) + e0_x3};
    real_t inv_gamma0{1.0 / std::sqrt(1.0 + u0_x1 * u0_x1 + u0_x2 * u0_x2 + u0_x3 * u0_x3)};

    COEFF *= inv_gamma0;
    b0_x1 *= COEFF;
    b0_x2 *= COEFF;
    b0_x3 *= COEFF;
    COEFF = 2.0 / (1.0 + b0_x1 * b0_x1 + b0_x2 * b0_x2 + b0_x3 * b0_x3);
    real_t u1_x1{(u0_x1 + u0_x2 * b0_x3 - u0_x3 * b0_x2) * COEFF};
    real_t u1_x2{(u0_x2 + u0_x3 * b0_x1 - u0_x1 * b0_x3) * COEFF};
    real_t u1_x3{(u0_x3 + u0_x1 * b0_x2 - u0_x2 * b0_x1) * COEFF};

    u0_x1 += u1_x2 * b0_x3 - u1_x3 * b0_x2 + e0_x1;
    u0_x2 += u1_x3 * b0_x1 - u1_x1 * b0_x3 + e0_x2;
    u0_x3 += u1_x1 * b0_x2 - u1_x2 * b0_x1 + e0_x3;

    m_particles.m_ux1(p) = u0_x1;
    m_particles.m_ux2(p) = u0_x2;
    m_particles.m_ux3(p) = u0_x3;

    inv_gamma0 = 1.0 / std::sqrt(1.0 + u0_x1 * u0_x1 + u0_x2 * u0_x2 + u0_x3 * u0_x3);
    m_particles.m_x1(p) += dt * m_particles.m_ux1(p) * inv_gamma0;
  }
};

class Boris2D : public Pusher<TWO_D> {
public:
  Boris2D(const Meshblock<TWO_D>& m_meshblock_,
          const Particles<TWO_D>& m_particles_,
          const real_t& coeff_,
          const real_t& dt_)
      : Pusher<TWO_D>{m_meshblock_, m_particles_, coeff_, dt_} {}
  Inline void operator()(const index_t p) const {
    // dummy fields
    real_t e0_x1{0.0}, e0_x2{0.1}, e0_x3{0.0};
    real_t b0_x1{0.0}, b0_x2{0.0}, b0_x3{1.0};
    real_t COEFF{coeff};

    e0_x1 *= COEFF;
    e0_x2 *= COEFF;
    e0_x3 *= COEFF;

    real_t u0_x1{m_particles.m_ux1(p) + e0_x1};
    real_t u0_x2{m_particles.m_ux2(p) + e0_x2};
    real_t u0_x3{m_particles.m_ux3(p) + e0_x3};
    real_t inv_gamma0{1.0 / std::sqrt(1.0 + u0_x1 * u0_x1 + u0_x2 * u0_x2 + u0_x3 * u0_x3)};

    COEFF *= inv_gamma0;
    b0_x1 *= COEFF;
    b0_x2 *= COEFF;
    b0_x3 *= COEFF;
    COEFF = 2.0 / (1.0 + b0_x1 * b0_x1 + b0_x2 * b0_x2 + b0_x3 * b0_x3);
    real_t u1_x1{(u0_x1 + u0_x2 * b0_x3 - u0_x3 * b0_x2) * COEFF};
    real_t u1_x2{(u0_x2 + u0_x3 * b0_x1 - u0_x1 * b0_x3) * COEFF};
    real_t u1_x3{(u0_x3 + u0_x1 * b0_x2 - u0_x2 * b0_x1) * COEFF};

    u0_x1 += u1_x2 * b0_x3 - u1_x3 * b0_x2 + e0_x1;
    u0_x2 += u1_x3 * b0_x1 - u1_x1 * b0_x3 + e0_x2;
    u0_x3 += u1_x1 * b0_x2 - u1_x2 * b0_x1 + e0_x3;

    m_particles.m_ux1(p) = u0_x1;
    m_particles.m_ux2(p) = u0_x2;
    m_particles.m_ux3(p) = u0_x3;

    inv_gamma0 = 1.0 / std::sqrt(1.0 + u0_x1 * u0_x1 + u0_x2 * u0_x2 + u0_x3 * u0_x3);
    m_particles.m_x1(p) += dt * m_particles.m_ux1(p) * inv_gamma0;
    m_particles.m_x2(p) += dt * m_particles.m_ux2(p) * inv_gamma0;
  }
};

class Boris3D : public Pusher<THREE_D> {
public:
  Boris3D(const Meshblock<THREE_D>& m_meshblock_,
          const Particles<THREE_D>& m_particles_,
          const real_t& coeff_,
          const real_t& dt_)
      : Pusher<THREE_D>{m_meshblock_, m_particles_, coeff_, dt_} {}
  Inline void operator()(const index_t p) const {
    // dummy fields
    real_t e0_x1{0.0}, e0_x2{1.0}, e0_x3{0.0};
    real_t b0_x1{0.0}, b0_x2{0.0}, b0_x3{1.0};
    real_t COEFF{coeff};

    e0_x1 *= COEFF;
    e0_x2 *= COEFF;
    e0_x3 *= COEFF;

    real_t u0_x1{m_particles.m_ux1(p) + e0_x1};
    real_t u0_x2{m_particles.m_ux2(p) + e0_x2};
    real_t u0_x3{m_particles.m_ux3(p) + e0_x3};
    real_t inv_gamma0{1.0 / std::sqrt(1.0 + u0_x1 * u0_x1 + u0_x2 * u0_x2 + u0_x3 * u0_x3)};

    COEFF *= inv_gamma0;
    b0_x1 *= COEFF;
    b0_x2 *= COEFF;
    b0_x3 *= COEFF;
    COEFF = 2.0 / (1.0 + b0_x1 * b0_x1 + b0_x2 * b0_x2 + b0_x3 * b0_x3);
    real_t u1_x1{(u0_x1 + u0_x2 * b0_x3 - u0_x3 * b0_x2) * COEFF};
    real_t u1_x2{(u0_x2 + u0_x3 * b0_x1 - u0_x1 * b0_x3) * COEFF};
    real_t u1_x3{(u0_x3 + u0_x1 * b0_x2 - u0_x2 * b0_x1) * COEFF};

    u0_x1 += u1_x2 * b0_x3 - u1_x3 * b0_x2 + e0_x1;
    u0_x2 += u1_x3 * b0_x1 - u1_x1 * b0_x3 + e0_x2;
    u0_x3 += u1_x1 * b0_x2 - u1_x2 * b0_x1 + e0_x3;

    m_particles.m_ux1(p) = u0_x1;
    m_particles.m_ux2(p) = u0_x2;
    m_particles.m_ux3(p) = u0_x3;

    inv_gamma0 = 1.0 / std::sqrt(1.0 + u0_x1 * u0_x1 + u0_x2 * u0_x2 + u0_x3 * u0_x3);
    m_particles.m_x1(p) += dt * m_particles.m_ux1(p) * inv_gamma0;
    m_particles.m_x2(p) += dt * m_particles.m_ux2(p) * inv_gamma0;
    m_particles.m_x3(p) += dt * m_particles.m_ux3(p) * inv_gamma0;
  }
};

} // namespace ntt

#endif
