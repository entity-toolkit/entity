#ifndef FRAMEWORK_PARTICLE_MACROS_H
#define FRAMEWORK_PARTICLE_MACROS_H

#define PRTL_X1(PARTICLES, P)                                                                 \
  (static_cast<real_t>((PARTICLES).i1((p))) + static_cast<real_t>((PARTICLES).dx1((p))))

#define PRTL_X2(PARTICLES, P)                                                                 \
  (static_cast<real_t>((PARTICLES).i2((p))) + static_cast<real_t>((PARTICLES).dx2((p))))

#define PRTL_X3(PARTICLES, P)                                                                 \
  (static_cast<real_t>((PARTICLES).i3((p))) + static_cast<real_t>((PARTICLES).dx3((p))))

#define PRTL_USQR_SR(PARTICLES, P)                                                            \
  (PARTICLES).ux1((P)) * (PARTICLES).ux1((P)) + (PARTICLES).ux2((P)) * (PARTICLES).ux2((P))   \
    + (PARTICLES).ux3((P)) * (PARTICLES).ux3((P))

#define PRTL_GAMMA_SR(PARTICLES, P)                                                           \
  math::sqrt(static_cast<real_t>(1.0) + PRTL_USQR_SR((PARTICLES), (P)))

#define PICPRTL_XYZ_1D(MBLOCK, SPECIES, INDEX, X1, U1, U2, U3)                                \
  {                                                                                           \
    using namespace ntt;                                                                      \
    coord_t<Dimension::ONE_D> X_CU;                                                           \
    ((MBLOCK)->metric).x_Cart2Code({(X1)}, X_CU);                                             \
    auto [I1, DX1]                              = ((MBLOCK)->metric).CU_to_Idi(X_CU[0]);      \
    (MBLOCK)->particles[(SPECIES)].i1((INDEX))  = I1;                                         \
    (MBLOCK)->particles[(SPECIES)].dx1((INDEX)) = DX1;                                        \
    (MBLOCK)->particles[(SPECIES)].ux1((INDEX)) = U1;                                         \
    (MBLOCK)->particles[(SPECIES)].ux2((INDEX)) = U2;                                         \
    (MBLOCK)->particles[(SPECIES)].ux3((INDEX)) = U3;                                         \
  }

#define PICPRTL_XYZ_2D(MBLOCK, SPECIES, INDEX, X1, X2, U1, U2, U3)                            \
  {                                                                                           \
    using namespace ntt;                                                                      \
    coord_t<Dimension::TWO_D> X_CU;                                                           \
    ((MBLOCK)->metric).x_Cart2Code({(X1), (X2)}, X_CU);                                       \
    auto [I1, DX1]                              = ((MBLOCK)->metric).CU_to_Idi(X_CU[0]);      \
    auto [I2, DX2]                              = ((MBLOCK)->metric).CU_to_Idi(X_CU[1]);      \
    (MBLOCK)->particles[(SPECIES)].i1((INDEX))  = I1;                                         \
    (MBLOCK)->particles[(SPECIES)].dx1((INDEX)) = DX1;                                        \
    (MBLOCK)->particles[(SPECIES)].i2((INDEX))  = I2;                                         \
    (MBLOCK)->particles[(SPECIES)].dx2((INDEX)) = DX2;                                        \
    (MBLOCK)->particles[(SPECIES)].ux1((INDEX)) = U1;                                         \
    (MBLOCK)->particles[(SPECIES)].ux2((INDEX)) = U2;                                         \
    (MBLOCK)->particles[(SPECIES)].ux3((INDEX)) = U3;                                         \
  }

#define PICPRTL_XYZ_3D(MBLOCK, SPECIES, INDEX, X1, X2, X3, U1, U2, U3)                        \
  {                                                                                           \
    using namespace ntt;                                                                      \
    coord_t<Dimension::THREE_D> X_CU;                                                         \
    ((MBLOCK)->metric).x_Cart2Code({(X1), (X2), (X3)}, X_CU);                                 \
    auto [I1, DX1]                              = ((MBLOCK)->metric).CU_to_Idi(X_CU[0]);      \
    auto [I2, DX2]                              = ((MBLOCK)->metric).CU_to_Idi(X_CU[1]);      \
    auto [I3, DX3]                              = ((MBLOCK)->metric).CU_to_Idi(X_CU[2]);      \
    (MBLOCK)->particles[(SPECIES)].i1((INDEX))  = I1;                                         \
    (MBLOCK)->particles[(SPECIES)].dx1((INDEX)) = DX1;                                        \
    (MBLOCK)->particles[(SPECIES)].i2((INDEX))  = I2;                                         \
    (MBLOCK)->particles[(SPECIES)].dx2((INDEX)) = DX2;                                        \
    (MBLOCK)->particles[(SPECIES)].i3((INDEX))  = I3;                                         \
    (MBLOCK)->particles[(SPECIES)].dx3((INDEX)) = DX3;                                        \
    (MBLOCK)->particles[(SPECIES)].ux1((INDEX)) = U1;                                         \
    (MBLOCK)->particles[(SPECIES)].ux2((INDEX)) = U2;                                         \
    (MBLOCK)->particles[(SPECIES)].ux3((INDEX)) = U3;                                         \
  }

#define PICPRTL_SPH_1D(MBLOCK, SPECIES, INDEX, X1, U1, U2, U3)                                \
  {                                                                                           \
    using namespace ntt;                                                                      \
    coord_t<Dimension::ONE_D> X_CU;                                                           \
    vec_t<Dimension::THREE_D> U_C {ZERO, ZERO, ZERO};                                         \
    ((MBLOCK)->metric)).x_Sph2Code({(X1)}, X_CU);                                             \
    ((MBLOCK)->metric)).v_Hat2Cart(X_CU, {U1, U2, U3}, U_C);                                  \
    auto [I1, DX1]                              = ((MBLOCK)->metric).CU_to_Idi(X_CU[0]);      \
    (MBLOCK)->particles[(SPECIES)].i1((INDEX))  = I1;                                         \
    (MBLOCK)->particles[(SPECIES)].dx1((INDEX)) = DX1;                                        \
    (MBLOCK)->particles[(SPECIES)].ux1((INDEX)) = U_C[0];                                     \
    (MBLOCK)->particles[(SPECIES)].ux2((INDEX)) = U_C[1];                                     \
    (MBLOCK)->particles[(SPECIES)].ux3((INDEX)) = U_C[2];                                     \
  }

#define PICPRTL_SPH_2D(MBLOCK, SPECIES, INDEX, X1, X2, U1, U2, U3)                            \
  {                                                                                           \
    using namespace ntt;                                                                      \
    coord_t<Dimension::TWO_D> X_CU;                                                           \
    vec_t<Dimension::THREE_D> U_C {ZERO, ZERO, ZERO};                                         \
    ((MBLOCK)->metric).x_Sph2Code({(X1), (X2)}, X_CU);                                        \
    ((MBLOCK)->metric).v_Hat2Cart({X_CU[0], X_CU[1], ZERO}, {U1, U2, U3}, U_C);               \
    auto [I1, DX1]                              = ((MBLOCK)->metric).CU_to_Idi(X_CU[0]);      \
    auto [I2, DX2]                              = ((MBLOCK)->metric).CU_to_Idi(X_CU[1]);      \
    (MBLOCK)->particles[(SPECIES)].i1((INDEX))  = I1;                                         \
    (MBLOCK)->particles[(SPECIES)].dx1((INDEX)) = DX1;                                        \
    (MBLOCK)->particles[(SPECIES)].i2((INDEX))  = I2;                                         \
    (MBLOCK)->particles[(SPECIES)].dx2((INDEX)) = DX2;                                        \
    (MBLOCK)->particles[(SPECIES)].ux1((INDEX)) = U_C[0];                                     \
    (MBLOCK)->particles[(SPECIES)].ux2((INDEX)) = U_C[1];                                     \
    (MBLOCK)->particles[(SPECIES)].ux3((INDEX)) = U_C[2];                                     \
  }

#define PICPRTL_SPH_3D(MBLOCK, SPECIES, INDEX, X1, X2, X3, U1, U2, U3)                        \
  {                                                                                           \
    using namespace ntt;                                                                      \
    coord_t<Dimension::THREE_D> X_CU;                                                         \
    vec_t<Dimension::THREE_D>   U_C {ZERO, ZERO, ZERO};                                       \
    ((MBLOCK)->metric).x_Sph2Code({(X1), (X2), (X3)}, X_CU);                                  \
    ((MBLOCK)->metric).v_Hat2Cart(X_CU, {U1, U2, U3}, U_C);                                   \
    auto [I1, DX1]                              = ((MBLOCK)->metric).CU_to_Idi(X_CU[0]);      \
    auto [I2, DX2]                              = ((MBLOCK)->metric).CU_to_Idi(X_CU[1]);      \
    auto [I3, DX3]                              = ((MBLOCK)->metric).CU_to_Idi(X_CU[2]);      \
    (MBLOCK)->particles[(SPECIES)].i1((INDEX))  = I1;                                         \
    (MBLOCK)->particles[(SPECIES)].dx1((INDEX)) = DX1;                                        \
    (MBLOCK)->particles[(SPECIES)].i2((INDEX))  = I2;                                         \
    (MBLOCK)->particles[(SPECIES)].dx2((INDEX)) = DX2;                                        \
    (MBLOCK)->particles[(SPECIES)].i3((INDEX))  = I3;                                         \
    (MBLOCK)->particles[(SPECIES)].dx3((INDEX)) = DX3;                                        \
    (MBLOCK)->particles[(SPECIES)].ux1((INDEX)) = U_C[0];                                     \
    (MBLOCK)->particles[(SPECIES)].ux2((INDEX)) = U_C[1];                                     \
    (MBLOCK)->particles[(SPECIES)].ux3((INDEX)) = U_C[2];                                     \
  }

#endif