#include "mathematics.h"

#include <algorithm>
#include <cmath>

namespace ntt {
  namespace math {
    constexpr double C_DOUBLE_ABS_EPSILON { 1e-12 };
    constexpr double C_DOUBLE_REL_EPSILON { 1e-8  };
    constexpr float  C_FLOAT_ABS_EPSILON  { 1e-6f };
    constexpr float  C_FLOAT_REL_EPSILON  { 1e-8f };
    namespace {
      bool numbersAreEqual(double a, double b, double absEpsilon, double relEpsilon) {
        double diff { std::abs(a - b) };
        if (diff <= absEpsilon)
          return true;
        a = std::abs(a); b = std::abs(b);
        double min = std::min(a, b);
        a -= min; b -= min;
        return (diff <= (std::max(std::abs(a), std::abs(b)) * relEpsilon));
      }
      bool numbersAreEqual(float a, float b, float absEpsilon, float relEpsilon) {
        float diff { std::abs(a - b) };
        if (diff <= absEpsilon)
          return true;
        a = std::abs(a); b = std::abs(b);
        float min = std::min(a, b);
        a -= min; b -= min;
        return (diff <= (std::max(std::abs(a), std::abs(b)) * relEpsilon));
      }
    }

    bool numbersAreEqual(double a, double b) {
      return numbersAreEqual(a, b, C_DOUBLE_ABS_EPSILON, C_DOUBLE_REL_EPSILON);
    }
    bool numbersAreEqual(float a, float b) {
      return numbersAreEqual(a, b, C_FLOAT_ABS_EPSILON, C_FLOAT_REL_EPSILON);
    }
  }
}
