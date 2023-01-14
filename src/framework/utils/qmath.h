#ifndef UTILS_QMATH_H
#define UTILS_QMATH_H

#include "wrapper.h"

#include <cfloat>

namespace ntt {

  /**
   * @brief Function to compare two float values.
   * @param a First value.
   * @param b Second value.
   * @param epsilon Accuracy.
   * @returns true/false.
   */
  Inline bool AlmostEqual(float a, float b, float epsilon = 0.00001f) {
    float absA = math::abs(a);
    float absB = math::abs(b);
    float diff = math::abs(a - b);

    if (a == b) {
      return true;
    } else if (a == 0.0 || b == 0.0 || (absA + absB < FLT_MIN)) {
      return diff < (epsilon * FLT_MIN);
    } else {
      return diff / math::min((absA + absB), FLT_MAX) < epsilon;
    }
  }

  /**
   * @brief Function to compare two double values.
   * @param a First value.
   * @param b Second value.
   * @param epsilon Accuracy.
   * @returns true/false.
   */
  Inline bool AlmostEqual(double a, double b, double epsilon = 1e-8) {
    double diff { math::abs(a - b) };
    if (diff <= 1e-12)
      return true;
    a = math::abs(a);
    b = math::abs(b);
    if (a == b) {
      return true;
    } else if (a == 0.0 || b == 0.0 || (a + b < 1e-12)) {
      return diff < (1e-12);
    } else {
      double min = math::min(a, b);
      a -= min;
      b -= min;
      return (diff <= (math::max(a, b) * epsilon));
    }
  }
}    // namespace ntt

#endif
