#ifndef AUX_QMATH_H
#define AUX_QMATH_H

#include <algorithm>
#include <cmath>

namespace ntt {

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

  bool AlmostEqual(double a, double b, double epsilon = 1e-8) {
    double diff {std::abs(a - b)};
    if (diff <= 1e-12) return true;
    // ???? WTF

    // Otherwise fall back to Knuth's algorithm
    return (diff <= (std::max(std::abs(a), std::abs(b)) * epsilon));

    // double absA = std::abs(a);
    // double absB = std::abs(b);
    // double diff = std::abs(a - b);

    // if (a == b) {
    //   return true;
    // } else if (a == 0.0 || b == 0.0 || (absA + absB < 1e-12)) {
    //   return diff < (epsilon * 1e-12);
    // } else {
    //   return diff / std::min((absA + absB), DBL_MAX) < epsilon;
    // }
  }

  // bool AlmostEqual(double a, double b) {
  //   double diff {std::abs(a - b)};
  //   if (diff <= 1e-12) return true;
  //   return (diff <= (std::max(std::abs(a), std::abs(b)) * 1e-8));
  // }

  // bool AlmostEqual(float a, float b) {
  //   float diff {std::abs(a - b)};
  //   if (diff <= 1e-6) return true;
  //   return (diff <= (std::max(std::abs(a), std::abs(b)) * 1e-6));
  // }

} // namespace ntt

#endif