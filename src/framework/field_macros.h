#ifndef FRAMEWORK_FIELD_MACROS_H
#define FRAMEWORK_FIELD_MACROS_H

#define GET_MACRO(_1, _2, _3, NAME, ...) NAME

#define BX1(...)                         GET_MACRO(__VA_ARGS__, BX1_3D, BX1_2D, BX1_1D, )(__VA_ARGS__)
#define BX1_1D(I)                        (m_mblock.em((I), em::bx1))
#define BX1_2D(I, J)                     (m_mblock.em((I), (J), em::bx1))
#define BX1_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::bx1))

#define BX2(...)                         GET_MACRO(__VA_ARGS__, BX2_3D, BX2_2D, BX2_1D, )(__VA_ARGS__)
#define BX2_1D(I)                        (m_mblock.em((I), em::bx2))
#define BX2_2D(I, J)                     (m_mblock.em((I), (J), em::bx2))
#define BX2_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::bx2))

#define BX3(...)                         GET_MACRO(__VA_ARGS__, BX3_3D, BX3_2D, BX3_1D, )(__VA_ARGS__)
#define BX3_1D(I)                        (m_mblock.em((I), em::bx3))
#define BX3_2D(I, J)                     (m_mblock.em((I), (J), em::bx3))
#define BX3_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::bx3))

#ifdef PIC_ENGINE

#  define EX1(...)        GET_MACRO(__VA_ARGS__, EX1_3D, EX1_2D, EX1_1D, )(__VA_ARGS__)
#  define EX1_1D(I)       (m_mblock.em((I), em::ex1))
#  define EX1_2D(I, J)    (m_mblock.em((I), (J), em::ex1))
#  define EX1_3D(I, J, K) (m_mblock.em((I), (J), (K), em::ex1))

#  define EX2(...)        GET_MACRO(__VA_ARGS__, EX2_3D, EX2_2D, EX2_1D, )(__VA_ARGS__)
#  define EX2_1D(I)       (m_mblock.em((I), em::ex2))
#  define EX2_2D(I, J)    (m_mblock.em((I), (J), em::ex2))
#  define EX2_3D(I, J, K) (m_mblock.em((I), (J), (K), em::ex2))

#  define EX3(...)        GET_MACRO(__VA_ARGS__, EX3_3D, EX3_2D, EX3_1D, )(__VA_ARGS__)
#  define EX3_1D(I)       (m_mblock.em((I), em::ex3))
#  define EX3_2D(I, J)    (m_mblock.em((I), (J), em::ex3))
#  define EX3_3D(I, J, K) (m_mblock.em((I), (J), (K), em::ex3))

#else

#  define DX1(...)         GET_MACRO(__VA_ARGS__, DX1_3D, DX1_2D, DX1_1D, )(__VA_ARGS__)
#  define DX1_1D(I)        (m_mblock.em((I), em::dx1))
#  define DX1_2D(I, J)     (m_mblock.em((I), (J), em::dx1))
#  define DX1_3D(I, J, K)  (m_mblock.em((I), (J), (K), em::dx1))

#  define DX2(...)         GET_MACRO(__VA_ARGS__, DX2_3D, DX2_2D, DX2_1D, )(__VA_ARGS__)
#  define DX2_1D(I)        (m_mblock.em((I), em::dx2))
#  define DX2_2D(I, J)     (m_mblock.em((I), (J), em::dx2))
#  define DX2_3D(I, J, K)  (m_mblock.em((I), (J), (K), em::dx2))

#  define DX3(...)         GET_MACRO(__VA_ARGS__, DX3_3D, DX3_2D, DX3_1D, )(__VA_ARGS__)
#  define DX3_1D(I)        (m_mblock.em((I), em::dx3))
#  define DX3_2D(I, J)     (m_mblock.em((I), (J), em::dx3))
#  define DX3_3D(I, J, K)  (m_mblock.em((I), (J), (K), em::dx3))

#  define B0X1(...)        GET_MACRO(__VA_ARGS__, B0X1_3D, B0X1_2D, B0X1_1D, )(__VA_ARGS__)
#  define B0X1_1D(I)       (m_mblock.em0((I), em::bx1))
#  define B0X1_2D(I, J)    (m_mblock.em0((I), (J), em::bx1))
#  define B0X1_3D(I, J, K) (m_mblock.em0((I), (J), (K), em::bx1))

#  define B0X2(...)        GET_MACRO(__VA_ARGS__, B0X2_3D, B0X2_2D, B0X2_1D, )(__VA_ARGS__)
#  define B0X2_1D(I)       (m_mblock.em0((I), em::bx2))
#  define B0X2_2D(I, J)    (m_mblock.em0((I), (J), em::bx2))
#  define B0X2_3D(I, J, K) (m_mblock.em0((I), (J), (K), em::bx2))

#  define B0X3(...)        GET_MACRO(__VA_ARGS__, B0X3_3D, B0X3_2D, B0X3_1D, )(__VA_ARGS__)
#  define B0X3_1D(I)       (m_mblock.em0((I), em::bx3))
#  define B0X3_2D(I, J)    (m_mblock.em0((I), (J), em::bx3))
#  define B0X3_3D(I, J, K) (m_mblock.em0((I), (J), (K), em::bx3))

#  define D0X1(...)        GET_MACRO(__VA_ARGS__, D0X1_3D, D0X1_2D, D0X1_1D, )(__VA_ARGS__)
#  define D0X1_1D(I)       (m_mblock.em0((I), em::dx1))
#  define D0X1_2D(I, J)    (m_mblock.em0((I), (J), em::dx1))
#  define D0X1_3D(I, J, K) (m_mblock.em0((I), (J), (K), em::dx1))

#  define D0X2(...)        GET_MACRO(__VA_ARGS__, D0X2_3D, D0X2_2D, D0X2_1D, )(__VA_ARGS__)
#  define D0X2_1D(I)       (m_mblock.em0((I), em::dx2))
#  define D0X2_2D(I, J)    (m_mblock.em0((I), (J), em::dx2))
#  define D0X2_3D(I, J, K) (m_mblock.em0((I), (J), (K), em::dx2))

#  define D0X3(...)        GET_MACRO(__VA_ARGS__, D0X3_3D, D0X3_2D, D0X3_1D, )(__VA_ARGS__)
#  define D0X3_1D(I)       (m_mblock.em0((I), em::dx3))
#  define D0X3_2D(I, J)    (m_mblock.em0((I), (J), em::dx3))
#  define D0X3_3D(I, J, K) (m_mblock.em0((I), (J), (K), em::dx3))

#  define HX1(...)         GET_MACRO(__VA_ARGS__, HX1_3D, HX1_2D, HX1_1D, )(__VA_ARGS__)
#  define HX1_1D(I)        (m_mblock.aux((I), em::hx1))
#  define HX1_2D(I, J)     (m_mblock.aux((I), (J), em::hx1))
#  define HX1_3D(I, J, K)  (m_mblock.aux((I), (J), (K), em::hx1))

#  define HX2(...)         GET_MACRO(__VA_ARGS__, HX2_3D, HX2_2D, HX2_1D, )(__VA_ARGS__)
#  define HX2_1D(I)        (m_mblock.aux((I), em::hx2))
#  define HX2_2D(I, J)     (m_mblock.aux((I), (J), em::hx2))
#  define HX2_3D(I, J, K)  (m_mblock.aux((I), (J), (K), em::hx2))

#  define HX3(...)         GET_MACRO(__VA_ARGS__, HX3_3D, HX3_2D, HX3_1D, )(__VA_ARGS__)
#  define HX3_1D(I)        (m_mblock.aux((I), em::hx3))
#  define HX3_2D(I, J)     (m_mblock.aux((I), (J), em::hx3))
#  define HX3_3D(I, J, K)  (m_mblock.aux((I), (J), (K), em::hx3))

#  define EX1(...)         GET_MACRO(__VA_ARGS__, EX1_3D, EX1_2D, EX1_1D, )(__VA_ARGS__)
#  define EX1_1D(I)        (m_mblock.aux((I), em::ex1))
#  define EX1_2D(I, J)     (m_mblock.aux((I), (J), em::ex1))
#  define EX1_3D(I, J, K)  (m_mblock.aux((I), (J), (K), em::ex1))

#  define EX2(...)         GET_MACRO(__VA_ARGS__, EX2_3D, EX2_2D, EX2_1D, )(__VA_ARGS__)
#  define EX2_1D(I)        (m_mblock.aux((I), em::ex2))
#  define EX2_2D(I, J)     (m_mblock.aux((I), (J), em::ex2))
#  define EX2_3D(I, J, K)  (m_mblock.aux((I), (J), (K), em::ex2))

#  define EX3(...)         GET_MACRO(__VA_ARGS__, EX3_3D, EX3_2D, EX3_1D, )(__VA_ARGS__)
#  define EX3_1D(I)        (m_mblock.aux((I), em::ex3))
#  define EX3_2D(I, J)     (m_mblock.aux((I), (J), em::ex3))
#  define EX3_3D(I, J, K)  (m_mblock.aux((I), (J), (K), em::ex3))

#endif

#define JX1(...)         GET_MACRO(__VA_ARGS__, JX1_3D, JX1_2D, JX1_1D, )(__VA_ARGS__)
#define JX1_1D(I)        (m_mblock.cur((I), cur::jx1))
#define JX1_2D(I, J)     (m_mblock.cur((I), (J), cur::jx1))
#define JX1_3D(I, J, K)  (m_mblock.cur((I), (J), (K), cur::jx1))

#define JX2(...)         GET_MACRO(__VA_ARGS__, JX2_3D, JX2_2D, JX2_1D, )(__VA_ARGS__)
#define JX2_1D(I)        (m_mblock.cur((I), cur::jx2))
#define JX2_2D(I, J)     (m_mblock.cur((I), (J), cur::jx2))
#define JX2_3D(I, J, K)  (m_mblock.cur((I), (J), (K), cur::jx2))

#define JX3(...)         GET_MACRO(__VA_ARGS__, JX3_3D, JX3_2D, JX3_1D, )(__VA_ARGS__)
#define JX3_1D(I)        (m_mblock.cur((I), cur::jx3))
#define JX3_2D(I, J)     (m_mblock.cur((I), (J), cur::jx3))
#define JX3_3D(I, J, K)  (m_mblock.cur((I), (J), (K), cur::jx3))

#define J0X1(...)        GET_MACRO(__VA_ARGS__, J0X1_3D, J0X1_2D, J0X1_1D, )(__VA_ARGS__)
#define J0X1_1D(I)       (m_mblock.cur0((I), cur::jx1))
#define J0X1_2D(I, J)    (m_mblock.cur0((I), (J), cur::jx1))
#define J0X1_3D(I, J, K) (m_mblock.cur0((I), (J), (K), cur::jx1))

#define J0X2(...)        GET_MACRO(__VA_ARGS__, J0X2_3D, J0X2_2D, J0X2_1D, )(__VA_ARGS__)
#define J0X2_1D(I)       (m_mblock.cur0((I), cur::jx2))
#define J0X2_2D(I, J)    (m_mblock.cur0((I), (J), cur::jx2))
#define J0X2_3D(I, J, K) (m_mblock.cur0((I), (J), (K), cur::jx2))

#define J0X3(...)        GET_MACRO(__VA_ARGS__, JX3_3D, JX3_2D, JX3_1D, )(__VA_ARGS__)
#define J0X3_1D(I)       (m_mblock.cur0((I), cur::jx3))
#define J0X3_2D(I, J)    (m_mblock.cur0((I), (J), cur::jx3))
#define J0X3_3D(I, J, K) (m_mblock.cur0((I), (J), (K), cur::jx3))

#define ATOMIC_JX1(...)                                                                       \
  GET_MACRO(__VA_ARGS__, ATOMIC_JX1_3D, ATOMIC_JX1_2D, ATOMIC_JX1_1D, )(__VA_ARGS__)
#define ATOMIC_JX1_1D(I)    (cur_access((I) + N_GHOSTS, cur::jx1))
#define ATOMIC_JX1_2D(I, J) (cur_access((I) + N_GHOSTS, (J) + N_GHOSTS, cur::jx1))
#define ATOMIC_JX1_3D(I, J, K)                                                                \
  (cur_access((I) + N_GHOSTS, (J) + N_GHOSTS, (K) + N_GHOSTS, cur::jx1))

#define ATOMIC_JX2(...)                                                                       \
  GET_MACRO(__VA_ARGS__, ATOMIC_JX2_3D, ATOMIC_JX2_2D, ATOMIC_JX2_1D, )(__VA_ARGS__)
#define ATOMIC_JX2_1D(I)    (cur_access((I) + N_GHOSTS, cur::jx2))
#define ATOMIC_JX2_2D(I, J) (cur_access((I) + N_GHOSTS, (J) + N_GHOSTS, cur::jx2))
#define ATOMIC_JX2_3D(I, J, K)                                                                \
  (cur_access((I) + N_GHOSTS, (J) + N_GHOSTS, (K) + N_GHOSTS, cur::jx2))

#define ATOMIC_JX3(...)                                                                       \
  GET_MACRO(__VA_ARGS__, ATOMIC_JX3_3D, ATOMIC_JX3_2D, ATOMIC_JX3_1D, )(__VA_ARGS__)
#define ATOMIC_JX3_1D(I)    (cur_access((I) + N_GHOSTS, cur::jx3))
#define ATOMIC_JX3_2D(I, J) (cur_access((I) + N_GHOSTS, (J) + N_GHOSTS, cur::jx3))
#define ATOMIC_JX3_3D(I, J, K)                                                                \
  (cur_access((I) + N_GHOSTS, (J) + N_GHOSTS, (K) + N_GHOSTS, cur::jx3))

#ifdef MINKOWSKI_METRIC

#  define set_em_E_2d(MBLOCK, I, J, XCODE, COMP, COMPI, FUNC, ...)                            \
    {                                                                                         \
      vec_t<Dim3>   e_hat { ZERO }, b_hat { ZERO };                                           \
      vec_t<Dim3>   e_cntrv { ZERO };                                                         \
      coord_t<Dim2> x_ph { ZERO };                                                            \
      (MBLOCK).metric.x_Code2Cart(x_code, x_ph);                                              \
      FUNC(x_ph, e_hat, b_hat, __VA_ARGS__);                                                  \
      (MBLOCK).metric.v3_Hat2Cntrv(x_code, e_hat, e_cntrv);                                   \
      (MBLOCK).em((I), (J), COMP) = e_cntrv[COMPI];                                           \
    }

#  define set_em_B_2d(MBLOCK, I, J, XCODE, COMP, COMPI, FUNC, ...)                            \
    {                                                                                         \
      vec_t<Dim3>   e_hat { ZERO }, b_hat { ZERO };                                           \
      vec_t<Dim3>   b_cntrv { ZERO };                                                         \
      coord_t<Dim2> x_ph { ZERO };                                                            \
      (MBLOCK).metric.x_Code2Cart(x_code, x_ph);                                              \
      FUNC(x_ph, e_hat, b_hat, __VA_ARGS__);                                                  \
      (MBLOCK).metric.v3_Hat2Cntrv(x_code, b_hat, b_cntrv);                                   \
      (MBLOCK).em((I), (J), COMP) = b_cntrv[COMPI];                                           \
    }

#else

#  define set_em_E_2d(MBLOCK, I, J, XCODE, COMP, COMPI, FUNC, ...)                            \
    {                                                                                         \
      vec_t<Dim3>   e_hat { ZERO }, b_hat { ZERO };                                           \
      vec_t<Dim3>   e_cntrv { ZERO };                                                         \
      coord_t<Dim2> x_ph { ZERO };                                                            \
      (MBLOCK).metric.x_Code2Sph(x_code, x_ph);                                               \
      FUNC(x_ph, e_hat, b_hat, __VA_ARGS__);                                                  \
      (MBLOCK).metric.v3_Hat2Cntrv(x_code, e_hat, e_cntrv);                                   \
      (MBLOCK).em((I), (J), COMP) = e_cntrv[COMPI];                                           \
    }

#  define set_em_B_2d(MBLOCK, I, J, XCODE, COMP, COMPI, FUNC, ...)                            \
    {                                                                                         \
      vec_t<Dim3>   e_hat { ZERO }, b_hat { ZERO };                                           \
      vec_t<Dim3>   b_cntrv { ZERO };                                                         \
      coord_t<Dim2> x_ph { ZERO };                                                            \
      (MBLOCK).metric.x_Code2Sph(x_code, x_ph);                                               \
      FUNC(x_ph, e_hat, b_hat, __VA_ARGS__);                                                  \
      (MBLOCK).metric.v3_Hat2Cntrv(x_code, b_hat, b_cntrv);                                   \
      (MBLOCK).em((I), (J), COMP) = b_cntrv[COMPI];                                           \
    }

#endif

#define set_ex1_2d(MBLOCK, I, J, FUNC, ...)                                                   \
  {                                                                                           \
    const real_t i_ { static_cast<real_t>(static_cast<int>((I)) - N_GHOSTS) };                \
    const real_t j_ { static_cast<real_t>(static_cast<int>((J)) - N_GHOSTS) };                \
    {                                                                                         \
      coord_t<Dim2> x_code;                                                                   \
      x_code[0] = i_ + HALF;                                                                  \
      x_code[1] = j_;                                                                         \
      set_em_E_2d((MBLOCK), (I), (J), (x_code), (em::ex1), (0), (FUNC), __VA_ARGS__);         \
    }                                                                                         \
  }

#define set_ex2_2d(MBLOCK, I, J, FUNC, ...)                                                   \
  {                                                                                           \
    const real_t i_ { static_cast<real_t>(static_cast<int>((I)) - N_GHOSTS) };                \
    const real_t j_ { static_cast<real_t>(static_cast<int>((J)) - N_GHOSTS) };                \
    {                                                                                         \
      coord_t<Dim2> x_code;                                                                   \
      x_code[0] = i_;                                                                         \
      x_code[1] = j_ + HALF;                                                                  \
      set_em_E_2d((MBLOCK), (I), (J), (x_code), (em::ex2), (1), (FUNC), __VA_ARGS__);         \
    }                                                                                         \
  }

#define set_ex3_2d(MBLOCK, I, J, FUNC, ...)                                                   \
  {                                                                                           \
    const real_t i_ { static_cast<real_t>(static_cast<int>((I)) - N_GHOSTS) };                \
    const real_t j_ { static_cast<real_t>(static_cast<int>((J)) - N_GHOSTS) };                \
    {                                                                                         \
      coord_t<Dim2> x_code;                                                                   \
      x_code[0] = i_;                                                                         \
      x_code[1] = j_;                                                                         \
      set_em_E_2d((MBLOCK), (I), (J), (x_code), (em::ex3), (2), (FUNC), __VA_ARGS__);         \
    }                                                                                         \
  }

#define set_bx1_2d(MBLOCK, I, J, FUNC, ...)                                                   \
  {                                                                                           \
    const real_t i_ { static_cast<real_t>(static_cast<int>((I)) - N_GHOSTS) };                \
    const real_t j_ { static_cast<real_t>(static_cast<int>((J)) - N_GHOSTS) };                \
    {                                                                                         \
      coord_t<Dim2> x_code;                                                                   \
      x_code[0] = i_;                                                                         \
      x_code[1] = j_ + HALF;                                                                  \
      set_em_B_2d((MBLOCK), (I), (J), (x_code), (em::bx1), (0), (FUNC), __VA_ARGS__);         \
    }                                                                                         \
  }

#define set_bx2_2d(MBLOCK, I, J, FUNC, ...)                                                   \
  {                                                                                           \
    const real_t i_ { static_cast<real_t>(static_cast<int>((I)) - N_GHOSTS) };                \
    const real_t j_ { static_cast<real_t>(static_cast<int>((J)) - N_GHOSTS) };                \
    {                                                                                         \
      coord_t<Dim2> x_code;                                                                   \
      x_code[0] = i_ + HALF;                                                                  \
      x_code[1] = j_;                                                                         \
      set_em_B_2d((MBLOCK), (I), (J), (x_code), (em::bx2), (1), (FUNC), __VA_ARGS__);         \
    }                                                                                         \
  }

#define set_bx3_2d(MBLOCK, I, J, FUNC, ...)                                                   \
  {                                                                                           \
    const real_t i_ { static_cast<real_t>(static_cast<int>((I)) - N_GHOSTS) };                \
    const real_t j_ { static_cast<real_t>(static_cast<int>((J)) - N_GHOSTS) };                \
    {                                                                                         \
      coord_t<Dim2> x_code;                                                                   \
      x_code[0] = i_ + HALF;                                                                  \
      x_code[1] = j_ + HALF;                                                                  \
      set_em_B_2d((MBLOCK), (I), (J), (x_code), (em::bx3), (2), (FUNC), __VA_ARGS__);         \
    }                                                                                         \
  }

#define set_em_fields_2d(MBLOCK, I, J, FUNC, ...)                                             \
  {                                                                                           \
    set_ex1_2d((MBLOCK), (I), (J), (FUNC), __VA_ARGS__);                                      \
    set_ex2_2d((MBLOCK), (I), (J), (FUNC), __VA_ARGS__);                                      \
    set_ex3_2d((MBLOCK), (I), (J), (FUNC), __VA_ARGS__);                                      \
    set_bx1_2d((MBLOCK), (I), (J), (FUNC), __VA_ARGS__);                                      \
    set_bx2_2d((MBLOCK), (I), (J), (FUNC), __VA_ARGS__);                                      \
    set_bx3_2d((MBLOCK), (I), (J), (FUNC), __VA_ARGS__);                                      \
  }

// regex

// find: m_mblock.em\((.*?), em::bx(.*?)\)
// replace: BX$2($1)

// find: m_mblock.em\((.*?), em::ex(.*?)\)
// replace: EX$2($1)

// find: m_mblock.cur\((.*?), cur::jx(.*?)\)
// replace: JX$2($1)

#endif