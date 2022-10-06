#ifndef FRAMEWORK_PARTICLE_MACROS_H
#define FRAMEWORK_PARTICLE_MACROS_H

#define from_Xi_to_i(XI, I)                                                                   \
  { I = static_cast<int>((XI) + N_GHOSTS) - N_GHOSTS; }

#define from_Xi_to_i_di(XI, I, DI)                                                            \
  {                                                                                           \
    from_Xi_to_i((XI), (I));                                                                  \
    DI = static_cast<float>((XI)) - static_cast<float>(I);                                    \
  }

#define get_prtl_x1(PARTICLES, P)                                                             \
  (static_cast<real_t>((PARTICLES).i1((P))) + static_cast<real_t>((PARTICLES).dx1((P))))

#define get_prtl_x2(PARTICLES, P)                                                             \
  (static_cast<real_t>((PARTICLES).i2((P))) + static_cast<real_t>((PARTICLES).dx2((P))))

#define get_prtl_x3(PARTICLES, P)                                                             \
  (static_cast<real_t>((PARTICLES).i3((P))) + static_cast<real_t>((PARTICLES).dx3((P))))

#define get_prtl_Usqr_SR(PARTICLES, P)                                                        \
  (PARTICLES).ux1((P)) * (PARTICLES).ux1((P)) + (PARTICLES).ux2((P)) * (PARTICLES).ux2((P))   \
    + (PARTICLES).ux3((P)) * (PARTICLES).ux3((P))

#define get_prtl_Gamma_SR(PARTICLES, P)                                                       \
  math::sqrt(static_cast<real_t>(1.0) + get_prtl_Usqr_SR((PARTICLES), (P)))

#define init_prtl_1d_XYZ(MBLOCK, SPECIES, INDEX, X1, U1, U2, U3)                              \
  {                                                                                           \
    coord_t<Dim1> X_CU;                                                                       \
    int           I;                                                                          \
    float         DX;                                                                         \
    ((MBLOCK).metric).x_Cart2Code({(X1)}, X_CU);                                              \
    from_Xi_to_i_di(X_CU[0], I, DX);                                                          \
    (SPECIES).i1((INDEX))  = I;                                                               \
    (SPECIES).dx1((INDEX)) = DX;                                                              \
    (SPECIES).ux1((INDEX)) = U1;                                                              \
    (SPECIES).ux2((INDEX)) = U2;                                                              \
    (SPECIES).ux3((INDEX)) = U3;                                                              \
  }

#define init_prtl_2d_XYZ(MBLOCK, SPECIES, INDEX, X1, X2, U1, U2, U3)                          \
  {                                                                                           \
    coord_t<Dim2> X_CU;                                                                       \
    int           I;                                                                          \
    float         DX;                                                                         \
    ((MBLOCK).metric).x_Cart2Code({(X1), (X2)}, X_CU);                                        \
    from_Xi_to_i_di(X_CU[0], I, DX);                                                          \
    (SPECIES).i1((INDEX))  = I;                                                               \
    (SPECIES).dx1((INDEX)) = DX;                                                              \
    from_Xi_to_i_di(X_CU[1], I, DX);                                                          \
    (SPECIES).i2((INDEX))  = I;                                                               \
    (SPECIES).dx2((INDEX)) = DX;                                                              \
    (SPECIES).ux1((INDEX)) = U1;                                                              \
    (SPECIES).ux2((INDEX)) = U2;                                                              \
    (SPECIES).ux3((INDEX)) = U3;                                                              \
  }

#define init_prtl_3d_XYZ(MBLOCK, SPECIES, INDEX, X1, X2, X3, U1, U2, U3)                      \
  {                                                                                           \
    coord_t<Dim3> X_CU;                                                                       \
    int           I;                                                                          \
    float         DX;                                                                         \
    ((MBLOCK).metric).x_Cart2Code({(X1), (X2), (X3)}, X_CU);                                  \
    from_Xi_to_i_di(X_CU[0], I, DX);                                                          \
    (SPECIES).i1((INDEX))  = I;                                                               \
    (SPECIES).dx1((INDEX)) = DX;                                                              \
    from_Xi_to_i_di(X_CU[1], I, DX);                                                          \
    (SPECIES).i2((INDEX))  = I;                                                               \
    (SPECIES).dx2((INDEX)) = DX;                                                              \
    from_Xi_to_i_di(X_CU[2], I, DX);                                                          \
    (SPECIES).i3((INDEX))  = I;                                                               \
    (SPECIES).dx3((INDEX)) = DX;                                                              \
    (SPECIES).ux1((INDEX)) = U1;                                                              \
    (SPECIES).ux2((INDEX)) = U2;                                                              \
    (SPECIES).ux3((INDEX)) = U3;                                                              \
  }

#define init_prtl_2d_Sph(MBLOCK, SPECIES, INDEX, X1, X2, U1, U2, U3)                          \
  {                                                                                           \
    coord_t<Dim2> X_CU;                                                                       \
    vec_t<Dim3>   U_C {ZERO, ZERO, ZERO};                                                     \
    int           I;                                                                          \
    float         DX;                                                                         \
    ((MBLOCK).metric).x_Sph2Code({(X1), (X2)}, X_CU);                                         \
    ((MBLOCK).metric).v_Hat2Cart({X_CU[0], X_CU[1], ZERO}, {U1, U2, U3}, U_C);                \
    from_Xi_to_i_di(X_CU[0], I, DX);                                                          \
    (SPECIES).i1((INDEX))  = I;                                                               \
    (SPECIES).dx1((INDEX)) = DX;                                                              \
    from_Xi_to_i_di(X_CU[1], I, DX);                                                          \
    (SPECIES).i2((INDEX))  = I;                                                               \
    (SPECIES).dx2((INDEX)) = DX;                                                              \
    (SPECIES).ux1((INDEX)) = U_C[0];                                                          \
    (SPECIES).ux2((INDEX)) = U_C[1];                                                          \
    (SPECIES).ux3((INDEX)) = U_C[2];                                                          \
  }

#define init_prtl_3d_Sph(MBLOCK, SPECIES, INDEX, X1, X2, X3, U1, U2, U3)                      \
  {                                                                                           \
    coord_t<Dim3> X_CU;                                                                       \
    vec_t<Dim3>   U_C {ZERO, ZERO, ZERO};                                                     \
    int           I;                                                                          \
    float         DX;                                                                         \
    ((MBLOCK).metric).x_Sph2Code({(X1), (X2), (X3)}, X_CU);                                   \
    ((MBLOCK).metric).v_Hat2Cart(X_CU, {U1, U2, U3}, U_C);                                    \
    from_Xi_to_i_di(X_CU[0], I, DX);                                                          \
    (SPECIES).i1((INDEX))  = I;                                                               \
    (SPECIES).dx1((INDEX)) = DX;                                                              \
    from_Xi_to_i_di(X_CU[1], I, DX);                                                          \
    (SPECIES).i2((INDEX))  = I;                                                               \
    (SPECIES).dx2((INDEX)) = DX;                                                              \
    from_Xi_to_i_di(X_CU[2], I, DX);                                                          \
    (SPECIES).i3((INDEX))  = I;                                                               \
    (SPECIES).dx3((INDEX)) = DX;                                                              \
    (SPECIES).ux1((INDEX)) = U_C[0];                                                          \
    (SPECIES).ux2((INDEX)) = U_C[1];                                                          \
    (SPECIES).ux3((INDEX)) = U_C[2];                                                          \
  }

#endif