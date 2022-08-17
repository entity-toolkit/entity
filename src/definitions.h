#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define STRINGIZE(x)        STRINGIZE_DETAIL(x)
#define STRINGIZE_DETAIL(x) #x
#define LINE_STRING         STRINGIZE(__LINE__)

#define MINKOWSKI_METRIC    1
#define SPHERICAL_METRIC    2
#define QSPHERICAL_METRIC   3
#define KERR_SCHILD_METRIC  4
#define QKERR_SCHILD_METRIC 5

#define PIC_SIMTYPE         1
#define GRPIC_SIMTYPE       2

// Defining precision-based constants and types
#ifdef SINGLE_PRECISION
using real_t = float;
#else
using real_t = double;
#endif

// Math constants
namespace ntt {
  namespace constant {
    inline constexpr double PI {3.14159265358979323846};
    inline constexpr double INV_PI {0.31830988618379067154};
    inline constexpr double PI_SQR {9.86960440108935861882};
    inline constexpr double INV_PI_SQR {0.10132118364233777144};
    inline constexpr double TWO_PI {6.28318530717958647692};
    inline constexpr double E {2.71828182845904523536};
    inline constexpr double SQRT2 {1.41421356237309504880};
    inline constexpr double INV_SQRT2 {0.70710678118654752440};
    inline constexpr double SQRT3 {1.73205080756887729352};
  } // namespace constant
} // namespace ntt

// Simple math expressions (for real-type arguments)
#ifdef SINGLE_PRECISION
inline constexpr float ONE {1.0f};
inline constexpr float TWO {2.0f};
inline constexpr float ZERO {0.0f};
inline constexpr float HALF {0.5f};
inline constexpr float QUARTER {0.25f};
#else
inline constexpr double ONE {1.0};
inline constexpr double TWO {2.0};
inline constexpr double ZERO {0.0};
inline constexpr double HALF {0.5};
inline constexpr double QUARTER {0.25};
#endif

#define IMIN(a, b)   ((a) < (b) ? (a) : (b))
#define IMAX(a, b)   ((a) > (b) ? (a) : (b))
#define SIGNd(x)     (((x) < 0.0) ? -1.0 : 1.0)
#define SIGNf(x)     (((x) < 0.0f) ? -1.0f : 1.0f)
#define SIGN(x)      (((x) < ZERO) ? -ONE : ONE)
#define HEAVISIDE(x) (((x) <= ZERO) ? ZERO : ONE)
#define SQR(x)       ((x) * (x))
#define CUBE(x)      ((x) * (x) * (x))

#define GET_MACRO(_1, _2, _3, NAME, ...) NAME

#define BX1(...)                         GET_MACRO(__VA_ARGS__, BX1_3D, BX1_2D, BX1_1D)(__VA_ARGS__)
#define BX1_1D(I)                        (m_mblock.em((I), em::bx1))
#define BX1_2D(I, J)                     (m_mblock.em((I), (J), em::bx1))
#define BX1_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::bx1))

#define BX2(...)                         GET_MACRO(__VA_ARGS__, BX2_3D, BX2_2D, BX2_1D)(__VA_ARGS__)
#define BX2_1D(I)                        (m_mblock.em((I), em::bx2))
#define BX2_2D(I, J)                     (m_mblock.em((I), (J), em::bx2))
#define BX2_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::bx2))

#define BX3(...)                         GET_MACRO(__VA_ARGS__, BX3_3D, BX3_2D, BX3_1D)(__VA_ARGS__)
#define BX3_1D(I)                        (m_mblock.em((I), em::bx3))
#define BX3_2D(I, J)                     (m_mblock.em((I), (J), em::bx3))
#define BX3_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::bx3))

#define EX1(...)                         GET_MACRO(__VA_ARGS__, EX1_3D, EX1_2D, EX1_1D)(__VA_ARGS__)
#define EX1_1D(I)                        (m_mblock.em((I), em::ex1))
#define EX1_2D(I, J)                     (m_mblock.em((I), (J), em::ex1))
#define EX1_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::ex1))

#define EX2(...)                         GET_MACRO(__VA_ARGS__, EX2_3D, EX2_2D, EX2_1D)(__VA_ARGS__)
#define EX2_1D(I)                        (m_mblock.em((I), em::ex2))
#define EX2_2D(I, J)                     (m_mblock.em((I), (J), em::ex2))
#define EX2_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::ex2))

#define EX3(...)                         GET_MACRO(__VA_ARGS__, EX3_3D, EX3_2D, EX3_1D)(__VA_ARGS__)
#define EX3_1D(I)                        (m_mblock.em((I), em::ex3))
#define EX3_2D(I, J)                     (m_mblock.em((I), (J), em::ex3))
#define EX3_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::ex3))

#endif
