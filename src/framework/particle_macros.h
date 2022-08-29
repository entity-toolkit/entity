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

#define PICPRTL_XYZ_1D(S, P, X1, U1, U2, U3)                                                  \
  {                                                                                           \
    coord_t<Dimension::ONE_D> X_CU;                                                           \
    vec_t<Dimension::THREE_D> U_C;                                                            \
    mblock.metric.x_Cart2Code({(X1)}, X_CU);                                                  \
    auto [I1, DX1]                 = mblock.metric.CU_to_Idi(X_CU[0]);                        \
    mblock.particles[(S)].i1((P))  = I1;                                                      \
    mblock.particles[(S)].dx1((P)) = DX1;                                                     \
    mblock.metric.v_Hat2Cart(X_CU, {U1, U2, U3}, U_C);                                        \
    mblock.particles[(S)].ux1((P)) = U_C[0];                                                  \
    mblock.particles[(S)].ux2((P)) = U_C[1];                                                  \
    mblock.particles[(S)].ux3((P)) = U_C[2];                                                  \
  }

#define PICPRTL_XYZ_2D(S, P, X1, X2, U1, U2, U3)                                              \
  {                                                                                           \
    coord_t<Dimension::TWO_D> X_CU;                                                           \
    vec_t<Dimension::THREE_D> U_C;                                                            \
    mblock.metric.x_Cart2Code({(X1), (X2)}, X_CU);                                            \
    auto [I1, DX1]                 = mblock.metric.CU_to_Idi(X_CU[0]);                        \
    auto [I2, DX2]                 = mblock.metric.CU_to_Idi(X_CU[1]);                        \
    mblock.particles[(S)].i1((P))  = I1;                                                      \
    mblock.particles[(S)].dx1((P)) = DX1;                                                     \
    mblock.particles[(S)].i2((P))  = I2;                                                      \
    mblock.particles[(S)].dx2((P)) = DX2;                                                     \
    mblock.metric.v_Hat2Cart(X_CU, {U1, U2, U3}, U_C);                                        \
    mblock.particles[(S)].ux1((P)) = U_C[0];                                                  \
    mblock.particles[(S)].ux2((P)) = U_C[1];                                                  \
    mblock.particles[(S)].ux3((P)) = U_C[2];                                                  \
  }

#define PICPRTL_XYZ_3D(S, P, X1, X2, X3, U1, U2, U3)                                          \
  {                                                                                           \
    coord_t<Dimension::THREE_D> X_CU;                                                         \
    vec_t<Dimension::THREE_D>   U_C;                                                          \
    mblock.metric.x_Cart2Code({(X1), (X2), (X3)}, X_CU);                                      \
    auto [I1, DX1]                 = mblock.metric.CU_to_Idi(X_CU[0]);                        \
    auto [I2, DX2]                 = mblock.metric.CU_to_Idi(X_CU[1]);                        \
    auto [I3, DX3]                 = mblock.metric.CU_to_Idi(X_CU[2]);                        \
    mblock.particles[(S)].i1((P))  = I1;                                                      \
    mblock.particles[(S)].dx1((P)) = DX1;                                                     \
    mblock.particles[(S)].i2((P))  = I2;                                                      \
    mblock.particles[(S)].dx2((P)) = DX2;                                                     \
    mblock.particles[(S)].i3((P))  = I3;                                                      \
    mblock.particles[(S)].dx3((P)) = DX3;                                                     \
    mblock.metric.v_Hat2Cart(X_CU, {U1, U2, U3}, U_C);                                        \
    mblock.particles[(S)].ux1((P)) = U_C[0];                                                  \
    mblock.particles[(S)].ux2((P)) = U_C[1];                                                  \
    mblock.particles[(S)].ux3((P)) = U_C[2];                                                  \
  }

#define PICPRTL_SPH_1D(S, P, X1, U1, U2, U3)                                                  \
  {                                                                                           \
    coord_t<Dimension::ONE_D> X_CU;                                                           \
    vec_t<Dimension::THREE_D> U_C;                                                            \
    mblock.metric.x_Sph2Code({(X1)}, X_CU);                                                   \
    auto [I1, DX1]                 = mblock.metric.CU_to_Idi(X_CU[0]);                        \
    mblock.particles[(S)].i1((P))  = I1;                                                      \
    mblock.particles[(S)].dx1((P)) = DX1;                                                     \
    mblock.metric.v_Hat2Cart(X_CU, {U1, U2, U3}, U_C);                                        \
    mblock.particles[(S)].ux1((P)) = U_C[0];                                                  \
    mblock.particles[(S)].ux2((P)) = U_C[1];                                                  \
    mblock.particles[(S)].ux3((P)) = U_C[2];                                                  \
  }

#define PICPRTL_SPH_2D(S, P, X1, X2, U1, U2, U3)                                              \
  {                                                                                           \
    coord_t<Dimension::TWO_D> X_CU;                                                           \
    vec_t<Dimension::THREE_D> U_C;                                                            \
    mblock.metric.x_Sph2Code({(X1), (X2)}, X_CU);                                             \
    auto [I1, DX1]                 = mblock.metric.CU_to_Idi(X_CU[0]);                        \
    auto [I2, DX2]                 = mblock.metric.CU_to_Idi(X_CU[1]);                        \
    mblock.particles[(S)].i1((P))  = I1;                                                      \
    mblock.particles[(S)].dx1((P)) = DX1;                                                     \
    mblock.particles[(S)].i2((P))  = I2;                                                      \
    mblock.particles[(S)].dx2((P)) = DX2;                                                     \
    mblock.metric.v_Hat2Cart({X_CU[0], X_CU[1], ZERO}, {U1, U2, U3}, U_C);                    \
    mblock.particles[(S)].ux1((P)) = U_C[0];                                                  \
    mblock.particles[(S)].ux2((P)) = U_C[1];                                                  \
    mblock.particles[(S)].ux3((P)) = U_C[2];                                                  \
  }

#define PICPRTL_SPH_3D(S, P, X1, X2, X3, U1, U2, U3)                                          \
  {                                                                                           \
    coord_t<Dimension::THREE_D> X_CU;                                                         \
    vec_t<Dimension::THREE_D>   U_C;                                                          \
    mblock.metric.x_Sph2Code({(X1), (X2), (X3)}, X_CU);                                       \
    auto [I1, DX1]                 = mblock.metric.CU_to_Idi(X_CU[0]);                        \
    auto [I2, DX2]                 = mblock.metric.CU_to_Idi(X_CU[1]);                        \
    auto [I3, DX3]                 = mblock.metric.CU_to_Idi(X_CU[2]);                        \
    mblock.particles[(S)].i1((P))  = I1;                                                      \
    mblock.particles[(S)].dx1((P)) = DX1;                                                     \
    mblock.particles[(S)].i2((P))  = I2;                                                      \
    mblock.particles[(S)].dx2((P)) = DX2;                                                     \
    mblock.particles[(S)].i3((P))  = I3;                                                      \
    mblock.particles[(S)].dx3((P)) = DX3;                                                     \
    mblock.metric.v_Hat2Cart(X_CU, {U1, U2, U3}, U_C);                                        \
    mblock.particles[(S)].ux1((P)) = U_C[0];                                                  \
    mblock.particles[(S)].ux2((P)) = U_C[1];                                                  \
    mblock.particles[(S)].ux3((P)) = U_C[2];                                                  \
  }

#endif