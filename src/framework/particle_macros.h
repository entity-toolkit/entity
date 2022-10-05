#ifndef FRAMEWORK_PARTICLE_MACROS_H
#define FRAMEWORK_PARTICLE_MACROS_H

#define Xi_TO_i(XI, I)                                                                        \
  { I = static_cast<int>((XI) + N_GHOSTS) - N_GHOSTS; }

#define Xi_TO_i_di(XI, I, DI)                                                                 \
  {                                                                                           \
    Xi_TO_i((XI), (I));                                                                       \
    DI = static_cast<float>((XI)) - static_cast<float>(I);                                    \
  }

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
    coord_t<Dim1> X_CU;                                                                       \
    int           I;                                                                          \
    float         DX;                                                                         \
    ((MBLOCK).metric).x_Cart2Code({(X1)}, X_CU);                                              \
    Xi_TO_i_di(X_CU[0], I, DX);                                                               \
    (SPECIES).i1((INDEX))  = I;                                                               \
    (SPECIES).dx1((INDEX)) = DX;                                                              \
    (SPECIES).ux1((INDEX)) = U1;                                                              \
    (SPECIES).ux2((INDEX)) = U2;                                                              \
    (SPECIES).ux3((INDEX)) = U3;                                                              \
  }

#define PICPRTL_XYZ_2D(MBLOCK, SPECIES, INDEX, X1, X2, U1, U2, U3)                            \
  {                                                                                           \
    coord_t<Dim2> X_CU;                                                                       \
    int           I;                                                                          \
    float         DX;                                                                         \
    ((MBLOCK).metric).x_Cart2Code({(X1), (X2)}, X_CU);                                        \
    Xi_TO_i_di(X_CU[0], I, DX);                                                               \
    (SPECIES).i1((INDEX))  = I;                                                               \
    (SPECIES).dx1((INDEX)) = DX;                                                              \
    Xi_TO_i_di(X_CU[1], I, DX);                                                               \
    (SPECIES).i2((INDEX))  = I;                                                               \
    (SPECIES).dx2((INDEX)) = DX;                                                              \
    (SPECIES).ux1((INDEX)) = U1;                                                              \
    (SPECIES).ux2((INDEX)) = U2;                                                              \
    (SPECIES).ux3((INDEX)) = U3;                                                              \
  }

#define PICPRTL_XYZ_3D(MBLOCK, SPECIES, INDEX, X1, X2, X3, U1, U2, U3)                        \
  {                                                                                           \
    coord_t<Dim3> X_CU;                                                                       \
    int           I;                                                                          \
    float         DX;                                                                         \
    ((MBLOCK).metric).x_Cart2Code({(X1), (X2), (X3)}, X_CU);                                  \
    Xi_TO_i_di(X_CU[0], I, DX);                                                               \
    (SPECIES).i1((INDEX))  = I;                                                               \
    (SPECIES).dx1((INDEX)) = DX;                                                              \
    Xi_TO_i_di(X_CU[1], I, DX);                                                               \
    (SPECIES).i2((INDEX))  = I;                                                               \
    (SPECIES).dx2((INDEX)) = DX;                                                              \
    Xi_TO_i_di(X_CU[2], I, DX);                                                               \
    (SPECIES).i3((INDEX))  = I;                                                               \
    (SPECIES).dx3((INDEX)) = DX;                                                              \
    (SPECIES).ux1((INDEX)) = U1;                                                              \
    (SPECIES).ux2((INDEX)) = U2;                                                              \
    (SPECIES).ux3((INDEX)) = U3;                                                              \
  }

#define PICPRTL_SPH_1D(MBLOCK, SPECIES, INDEX, X1, U1, U2, U3)                                \
  {                                                                                           \
    coord_t<Dim1> X_CU;                                                                       \
    vec_t<Dim3>   U_C {ZERO, ZERO, ZERO};                                                     \
    int           I;                                                                          \
    float         DX;                                                                         \
    ((MBLOCK).metric)).x_Sph2Code({(X1)}, X_CU);                                              \
    ((MBLOCK).metric)).v_Hat2Cart(X_CU, {U1, U2, U3}, U_C);                                   \
    Xi_TO_i_di(X_CU[0], I, DX);                                                               \
    (SPECIES).i1((INDEX))  = I;                                                               \
    (SPECIES).dx1((INDEX)) = DX;                                                              \
    (SPECIES).ux1((INDEX)) = U_C[0];                                                          \
    (SPECIES).ux2((INDEX)) = U_C[1];                                                          \
    (SPECIES).ux3((INDEX)) = U_C[2];                                                          \
  }

#define PICPRTL_SPH_2D(MBLOCK, SPECIES, INDEX, X1, X2, U1, U2, U3)                            \
  {                                                                                           \
    coord_t<Dim2> X_CU;                                                                       \
    vec_t<Dim3>   U_C {ZERO, ZERO, ZERO};                                                     \
    int           I;                                                                          \
    float         DX;                                                                         \
    ((MBLOCK).metric).x_Sph2Code({(X1), (X2)}, X_CU);                                         \
    ((MBLOCK).metric).v_Hat2Cart({X_CU[0], X_CU[1], ZERO}, {U1, U2, U3}, U_C);                \
    Xi_TO_i_di(X_CU[0], I, DX);                                                               \
    (SPECIES).i1((INDEX))  = I;                                                               \
    (SPECIES).dx1((INDEX)) = DX;                                                              \
    Xi_TO_i_di(X_CU[1], I, DX);                                                               \
    (SPECIES).i2((INDEX))  = I;                                                               \
    (SPECIES).dx2((INDEX)) = DX;                                                              \
    (SPECIES).ux1((INDEX)) = U_C[0];                                                          \
    (SPECIES).ux2((INDEX)) = U_C[1];                                                          \
    (SPECIES).ux3((INDEX)) = U_C[2];                                                          \
  }

#define PICPRTL_SPH_3D(MBLOCK, SPECIES, INDEX, X1, X2, X3, U1, U2, U3)                        \
  {                                                                                           \
    coord_t<Dim3> X_CU;                                                                       \
    vec_t<Dim3>   U_C {ZERO, ZERO, ZERO};                                                     \
    int           I;                                                                          \
    float         DX;                                                                         \
    ((MBLOCK).metric).x_Sph2Code({(X1), (X2), (X3)}, X_CU);                                   \
    ((MBLOCK).metric).v_Hat2Cart(X_CU, {U1, U2, U3}, U_C);                                    \
    Xi_TO_i_di(X_CU[0], I, DX);                                                               \
    (SPECIES).i1((INDEX))  = I;                                                               \
    (SPECIES).dx1((INDEX)) = DX;                                                              \
    Xi_TO_i_di(X_CU[1], I, DX);                                                               \
    (SPECIES).i2((INDEX))  = I;                                                               \
    (SPECIES).dx2((INDEX)) = DX;                                                              \
    Xi_TO_i_di(X_CU[2], I, DX);                                                               \
    (SPECIES).i3((INDEX))  = I;                                                               \
    (SPECIES).dx3((INDEX)) = DX;                                                              \
    (SPECIES).ux1((INDEX)) = U_C[0];                                                          \
    (SPECIES).ux2((INDEX)) = U_C[1];                                                          \
    (SPECIES).ux3((INDEX)) = U_C[2];                                                          \
  }

#endif