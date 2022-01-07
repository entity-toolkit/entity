#ifndef DEFINITIONS_H
#define DEFINITIONS_H

/**
 * Math constants
 */
namespace ntt {
#ifdef SINGLE_PRECISION
  inline constexpr float ONE {1.0f};
  inline constexpr float ZERO {0.0f};
  inline constexpr float HALF {0.5f};
#else
  inline constexpr double ONE {1.0};
  inline constexpr double ZERO {0.0};
  inline constexpr double HALF {0.5};
#endif

  inline constexpr double PI {
    3.14159265358979323846
  };
  inline constexpr double INV_PI {
    0.31830988618379067154
  };
  inline constexpr double PI_SQR {
    9.86960440108935861882
  };
  inline constexpr double INV_PI_SQR {
    0.10132118364233777144
  };
  inline constexpr double TWO_PI {
    6.28318530717958647692
  };
  inline constexpr double E {
    2.71828182845904523536
  };
  inline constexpr double SQRT2 {
    1.41421356237309504880
  };
  inline constexpr double INV_SQRT2 {
    0.70710678118654752440
  };
  inline constexpr double SQRT3 {
    1.73205080756887729352
  };
} // namespace ntt

/**
 * Simple math expressions (for real-type argument)
 */
#define SIGN(x)      (((x) < ZERO) ? -ONE : ONE)
#define HEAVISIDE(x) (((x) <= ZERO) ? ZERO : ONE)
#define SQR(x)       ((x) * (x))
#define CUBE(x)      ((x) * (x) * (x))

#endif
