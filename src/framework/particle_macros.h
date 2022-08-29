#ifndef FRAMEWORK_PARTICLE_MACROS_H
#define FRAMEWORK_PARTICLE_MACROS_H

#define PRTL_X1(P)                                                                            \
  (static_cast<real_t>(m_particles.i1((p))) + static_cast<real_t>(m_particles.dx1((p))))

#define PRTL_X2(P)                                                                            \
  (static_cast<real_t>(m_particles.i2((p))) + static_cast<real_t>(m_particles.dx2((p))))

#define PRTL_X3(P)                                                                            \
  (static_cast<real_t>(m_particles.i3((p))) + static_cast<real_t>(m_particles.dx3((p))))

#define PRTL_USQR_SR(P)                                                                       \
  m_particles.ux1((P)) * m_particles.ux1((P)) + m_particles.ux2((P)) * m_particles.ux2((P))   \
    + m_particles.ux3((P)) * m_particles.ux3((P))

#define PRTL_GAMMA_SR(P) math::sqrt(static_cast<real_t>(1.0) + PRTL_USQR_SR((P)))

#endif