#ifndef FRAMEWORK_PARTICLE_MACROS_H
#define FRAMEWORK_PARTICLE_MACROS_H

#define from_Xi_to_i(XI, I)                                                    \
  { I = static_cast<int>((XI)); }

#define from_Xi_to_i_di(XI, I, DI)                                             \
  {                                                                            \
    from_Xi_to_i((XI), (I));                                                   \
    DI = static_cast<prtldx_t>((XI)) - static_cast<prtldx_t>(I);               \
  }

#define Xi_to_i(XI) static_cast<int>((XI))

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))

#define get_prtl_x1(PARTICLES, P)                                              \
  (static_cast<real_t>((PARTICLES).i1((P))) +                                  \
   static_cast<real_t>((PARTICLES).dx1((P))))

#define get_prtl_x2(PARTICLES, P)                                              \
  (static_cast<real_t>((PARTICLES).i2((P))) +                                  \
   static_cast<real_t>((PARTICLES).dx2((P))))

#define get_prtl_x3(PARTICLES, P)                                              \
  (static_cast<real_t>((PARTICLES).i3((P))) +                                  \
   static_cast<real_t>((PARTICLES).dx3((P))))

#define get_prtl_x1_prev(PARTICLES, P)                                         \
  (static_cast<real_t>((PARTICLES).i1_prev((P))) +                             \
   static_cast<real_t>((PARTICLES).dx1_prev((P))))
#define get_prtl_x2_prev(PARTICLES, P)                                         \
  (static_cast<real_t>((PARTICLES).i2_prev((P))) +                             \
   static_cast<real_t>((PARTICLES).dx2_prev((P))))
#define get_prtl_x3_prev(PARTICLES, P)                                         \
  (static_cast<real_t>((PARTICLES).i3_prev((P))) +                             \
   static_cast<real_t>((PARTICLES).dx3_prev((P))))

#define init_prtl_1d_i_di(SPECIES, INDEX, I1, DI1, U1, U2, U3, WEIGHT)         \
  {                                                                            \
    (SPECIES).i1((INDEX))     = I1;                                            \
    (SPECIES).dx1((INDEX))    = DI1;                                           \
    (SPECIES).ux1((INDEX))    = U1;                                            \
    (SPECIES).ux2((INDEX))    = U2;                                            \
    (SPECIES).ux3((INDEX))    = U3;                                            \
    (SPECIES).tag((INDEX))    = static_cast<short>(ParticleTag::alive);        \
    (SPECIES).weight((INDEX)) = WEIGHT;                                        \
  }

#define init_prtl_2d_i_di(SPECIES, INDEX, I1, I2, DI1, DI2, U1, U2, U3, WEIGHT) \
  {                                                                             \
    (SPECIES).i1((INDEX))     = I1;                                             \
    (SPECIES).dx1((INDEX))    = DI1;                                            \
    (SPECIES).i2((INDEX))     = I2;                                             \
    (SPECIES).dx2((INDEX))    = DI2;                                            \
    (SPECIES).ux1((INDEX))    = U1;                                             \
    (SPECIES).ux2((INDEX))    = U2;                                             \
    (SPECIES).ux3((INDEX))    = U3;                                             \
    (SPECIES).tag((INDEX))    = static_cast<short>(ParticleTag::alive);         \
    (SPECIES).weight((INDEX)) = WEIGHT;                                         \
  }

#define init_prtl_3d_i_di(SPECIES, INDEX, I1, I2, I3, DI1, DI2, DI3, U1, U2, U3, WEIGHT) \
  {                                                                                      \
    (SPECIES).i1((INDEX))     = I1;                                                      \
    (SPECIES).dx1((INDEX))    = DI1;                                                     \
    (SPECIES).i2((INDEX))     = I2;                                                      \
    (SPECIES).dx2((INDEX))    = DI2;                                                     \
    (SPECIES).i3((INDEX))     = I3;                                                      \
    (SPECIES).dx3((INDEX))    = DI3;                                                     \
    (SPECIES).ux1((INDEX))    = U1;                                                      \
    (SPECIES).ux2((INDEX))    = U2;                                                      \
    (SPECIES).ux3((INDEX))    = U3;                                                      \
    (SPECIES).tag((INDEX))    = static_cast<short>(ParticleTag::alive);                  \
    (SPECIES).weight((INDEX)) = WEIGHT;                                                  \
  }

#if defined(PIC_ENGINE)

  #define init_prtl_1d(MBLOCK, SPECIES, INDEX, X1, U1, U2, U3, WEIGHT)         \
    {                                                                          \
      coord_t<Dim1> X_CU;                                                      \
      int           I;                                                         \
      prtldx_t      DX;                                                        \
      ((MBLOCK).metric).x_Phys2Code({ (X1) }, X_CU);                           \
      from_Xi_to_i_di(X_CU[0], I, DX);                                         \
      init_prtl_1d_i_di(SPECIES, INDEX, I, DX, U1, U2, U3, WEIGHT);            \
    }

  #ifdef MINKOWSKI_METRIC
    #define init_prtl_2d(MBLOCK, SPECIES, INDEX, X1, X2, U1, U2, U3, WEIGHT)     \
      {                                                                          \
        coord_t<Dim2> X_CU;                                                      \
        int           I1, I2;                                                    \
        prtldx_t      DX1, DX2;                                                  \
        ((MBLOCK).metric).x_Phys2Code({ (X1), (X2) }, X_CU);                     \
        from_Xi_to_i_di(X_CU[0], I1, DX1);                                       \
        from_Xi_to_i_di(X_CU[1], I2, DX2);                                       \
        init_prtl_2d_i_di(SPECIES, INDEX, I1, I2, DX1, DX2, U1, U2, U3, WEIGHT); \
      }
  #else
    #define init_prtl_2d(MBLOCK, SPECIES, INDEX, X1, X2, U1, U2, U3, WEIGHT)                 \
      {                                                                                      \
        coord_t<Dim2> X_CU;                                                                  \
        vec_t<Dim3>   U_C { ZERO, ZERO, ZERO };                                              \
        int           I1, I2;                                                                \
        prtldx_t      DX1, DX2;                                                              \
        ((MBLOCK).metric).x_Phys2Code({ (X1), (X2) }, X_CU);                                 \
        ((MBLOCK).metric).v3_Hat2Cart({ X_CU[0], X_CU[1], ZERO }, { U1, U2, U3 }, U_C);      \
        from_Xi_to_i_di(X_CU[0], I1, DX1);                                                   \
        from_Xi_to_i_di(X_CU[1], I2, DX2);                                                   \
        init_prtl_2d_i_di(SPECIES, INDEX, I1, I2, DX1, DX2, U_C[0], U_C[1], U_C[2], WEIGHT); \
      }
  #endif

  #define init_prtl_3d(MBLOCK, SPECIES, INDEX, X1, X2, X3, U1, U2, U3, WEIGHT) \
    {                                                                          \
      coord_t<Dim3> X_CU;                                                      \
      vec_t<Dim3>   U_C { ZERO, ZERO, ZERO };                                  \
      int           I1, I2, I3;                                                \
      prtldx_t      DX1, DX2, DX3;                                             \
      ((MBLOCK).metric).x_Phys2Code({ (X1), (X2), (X3) }, X_CU);               \
      ((MBLOCK).metric).v3_Hat2Cart(X_CU, { U1, U2, U3 }, U_C);                \
      from_Xi_to_i_di(X_CU[0], I1, DX1);                                       \
      from_Xi_to_i_di(X_CU[1], I2, DX2);                                       \
      from_Xi_to_i_di(X_CU[2], I3, DX3);                                       \
      init_prtl_3d_i_di(SPECIES,                                               \
                        INDEX,                                                 \
                        I1,                                                    \
                        I2,                                                    \
                        I3,                                                    \
                        DX1,                                                   \
                        DX2,                                                   \
                        DX3,                                                   \
                        U_C[0],                                                \
                        U_C[1],                                                \
                        U_C[2],                                                \
                        WEIGHT);                                               \
    }

#elif defined(GRPIC_ENGINE)

  #define init_prtl_2d(MBLOCK, SPECIES, INDEX, X1, X2, U1, U2, U3, WEIGHT)                 \
    {                                                                                      \
      coord_t<Dim2> X_CU;                                                                  \
      vec_t<Dim3>   U_C { ZERO, ZERO, ZERO };                                              \
      int           I1, I2;                                                                \
      prtldx_t      DX1, DX2;                                                              \
      ((MBLOCK).metric).x_Phys2Code({ (X1), (X2) }, X_CU);                                 \
      ((MBLOCK).metric).v3_Hat2Cov({ X_CU[0], X_CU[1] }, { U1, U2, U3 }, U_C);             \
      from_Xi_to_i_di(X_CU[0], I1, DX1);                                                   \
      from_Xi_to_i_di(X_CU[1], I2, DX2);                                                   \
      init_prtl_2d_i_di(SPECIES, INDEX, I1, I2, DX1, DX2, U_C[0], U_C[1], U_C[2], WEIGHT); \
    }

  #define init_prtl_2d_covariant(MBLOCK, SPECIES, INDEX, X1, X2, U1, U2, U3, WEIGHT)       \
    {                                                                                      \
      coord_t<Dim2> X_CU;                                                                  \
      vec_t<Dim3>   U_C { ZERO, ZERO, ZERO };                                              \
      int           I1, I2;                                                                \
      prtldx_t      DX1, DX2;                                                              \
      ((MBLOCK).metric).x_Phys2Code({ (X1), (X2) }, X_CU);                                 \
      ((MBLOCK).metric).v3_PhysCov2Cov({ X_CU[0], X_CU[1] }, { U1, U2, U3 }, U_C);         \
      from_Xi_to_i_di(X_CU[0], I1, DX1);                                                   \
      from_Xi_to_i_di(X_CU[1], I2, DX2);                                                   \
      init_prtl_2d_i_di(SPECIES, INDEX, I1, I2, DX1, DX2, U_C[0], U_C[1], U_C[2], WEIGHT); \
    }

#endif

#define InjectParticle_1D(MBLOCK, SPECIES, ATOMIC_INDEX, OFFSET, X1, U1, U2, U3, WEIGHT) \
  {                                                                                      \
    if ((X1) >= (MBLOCK).metric.x1_min && (X1) < (MBLOCK).metric.x1_max) {               \
      auto p { (OFFSET) + Kokkos::atomic_fetch_add(&(ATOMIC_INDEX)(), 1) };              \
      init_prtl_1d(MBLOCK, SPECIES, p, X1, U1, U2, U3, WEIGHT);                          \
    }                                                                                    \
  }
#define InjectParticle_2D(MBLOCK, SPECIES, ATOMIC_INDEX, OFFSET, X1, X2, U1, U2, U3, WEIGHT) \
  {                                                                                          \
    if ((X1) >= (MBLOCK).metric.x1_min && (X1) < (MBLOCK).metric.x1_max &&                   \
        (X2) >= (MBLOCK).metric.x2_min && (X2) < (MBLOCK).metric.x2_max) {                   \
      auto P { (OFFSET) + Kokkos::atomic_fetch_add(&(ATOMIC_INDEX)(), 1) };                  \
      init_prtl_2d(MBLOCK, SPECIES, P, X1, X2, U1, U2, U3, WEIGHT);                          \
    }                                                                                        \
  }
#define InjectParticle_3D(MBLOCK, SPECIES, ATOMIC_INDEX, OFFSET, X1, X2, X3, U1, U2, U3, WEIGHT) \
  {                                                                                              \
    if ((X1) >= (MBLOCK).metric.x1_min && (X1) < (MBLOCK).metric.x1_max &&                       \
        (X2) >= (MBLOCK).metric.x2_min && (X2) < (MBLOCK).metric.x2_max &&                       \
        (X3) >= (MBLOCK).metric.x3_min && (X3) < (MBLOCK).metric.x3_max) {                       \
      auto p { (OFFSET) + Kokkos::atomic_fetch_add(&(ATOMIC_INDEX)(), 1) };                      \
      init_prtl_3d(MBLOCK, SPECIES, p, X1, X2, X3, U1, U2, U3, WEIGHT);                          \
    }                                                                                            \
  }

#endif