#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define STRINGIZE(x)        STRINGIZE_DETAIL(x)
#define STRINGIZE_DETAIL(x) #x
#define LINE_STRING         STRINGIZE(__LINE__)

#define MINKOWSKI_METRIC    1
#define SPHERICAL_METRIC    2
#define QSPHERICAL_METRIC   3
#define KERR_SCHILD_METRIC   4

#define PIC_SIMTYPE         1
#define GRPIC_SIMTYPE       2

#define NTTError(msg)       throw std::runtime_error("# ERROR: " msg " : filename: " __FILE__ " : line: " LINE_STRING)

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
  } // namespace const
} // namespace ntt

// Simple math expressions (for real-type arguments)
#ifdef SINGLE_PRECISION
inline constexpr float ONE {1.0f};
inline constexpr float TWO {2.0f};
inline constexpr float ZERO {0.0f};
inline constexpr float HALF {0.5f};
#else
inline constexpr double ONE {1.0};
inline constexpr double TWO {2.0};
inline constexpr double ZERO {0.0};
inline constexpr double HALF {0.5};
#endif

#define SIGN(x)      (((x) < ZERO) ? -ONE : ONE)
#define HEAVISIDE(x) (((x) <= ZERO) ? ZERO : ONE)
#define SQR(x)       ((x) * (x))
#define CUBE(x)      ((x) * (x) * (x))

#endif
