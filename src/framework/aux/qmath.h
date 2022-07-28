#ifndef AUX_QMATH_H
#define AUX_QMATH_H

#include <algorithm>
#include <cmath>

namespace ntt {

  /**
   * @brief Function to compare two float values.
   * @param a First value.
   * @param b Second value.
   * @param epsilon Accuracy.
   * @returns true/false.
   */
  bool AlmostEqual(float a, float b, float epsilon = 0.00001f) {
    float absA = std::abs(a);
    float absB = std::abs(b);
    float diff = std::abs(a - b);

    if (a == b) {
      return true;
    } else if (a == 0.0 || b == 0.0 || (absA + absB < FLT_MIN)) {
      return diff < (epsilon * FLT_MIN);
    } else {
      return diff / std::min((absA + absB), FLT_MAX) < epsilon;
    }
  }

  /**
   * @brief Function to compare two double values.
   * @param a First value.
   * @param b Second value.
   * @param epsilon Accuracy.
   * @returns true/false.
   */
  bool AlmostEqual(double a, double b, double epsilon = 1e-8) {
    double diff {std::abs(a - b)};
    if (diff <= 1e-12) return true;
    a = std::abs(a);
    b = std::abs(b);
    if (a == b) {
      return true;
    } else if (a == 0.0 || b == 0.0 || (a + b < 1e-12)) {
      return diff < (1e-12);
    } else {
      double min = std::min(a, b);
      a -= min;
      b -= min;
      return (diff <= (std::max(a, b) * epsilon));
    }
  }
} // namespace ntt

#endif
