#include "qmath.h"

#include <algorithm>
#include <cmath>
#include <cfloat>

namespace ntt {

  bool AlmostEqual(float a, float b, float epsilon) {
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

  bool AlmostEqual(double a, double b, double epsilon) {
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