#include "math.h"

#include <algorithm>
#include <cmath>

namespace math {
  constexpr double c_DoubleAbsEpsilon { 1e-12 };
  constexpr double c_DoubleRelEpsilon { 1e-8  };
  constexpr float  c_FloatAbsEpsilon  { 1e-6f };
  constexpr float  c_FloatRelEpsilon  { 1e-8f };
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
    return numbersAreEqual(a, b, c_DoubleAbsEpsilon, c_DoubleRelEpsilon);
  }
  bool numbersAreEqual(float a, float b) {
    return numbersAreEqual(a, b, c_FloatAbsEpsilon, c_FloatRelEpsilon);
  }
}
