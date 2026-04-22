#include "utils/comparators.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

template <class T>
auto ulp_eq(T a, T b, std::size_t n = 1) -> bool {
  const T   m   = std::min(std::fabs(a), std::fabs(b));
  const int exp = m < std::numeric_limits<T>::min()
                    ? std::numeric_limits<T>::min_exponent - 1
                    : std::ilogb(m);
  return std::fabs(b - a) <= n * std::ldexp(std::numeric_limits<T>::epsilon(), exp);
}

auto main() -> int {
  {
    const auto f  = 1.0f;
    const auto f1 = 0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f +
                    0.1f + 0.1f;

    errorIf(ulp_eq(f, f1) != cmp::AlmostEqual(f, f1),
            "Wrong comparison with ULP-1 #f1");
    errorIf(ulp_eq(f, f1, 2) != cmp::AlmostEqual(f, f1, 1e-6f),
            "Wrong comparison with ULP-2 #f2");
  }

  {
    const auto f1 = 1e7f;
    const auto f2 = 9999999.0f;
    const auto df = 1.0f;

    errorIf(ulp_eq(f1 + df, f2 + df) && cmp::AlmostEqual(f1 + df, f2 + df, 1e-8f),
            "Wrong comparison with ULP-1 #f3");
    errorIf(ulp_eq(f1, f2) && cmp::AlmostEqual(f1, f2, 1e-8f),
            "Wrong comparison with ULP-1 #f4");
    errorIf(ulp_eq(f1, std::nexttoward(f1, +INFINITY)) &&
              cmp::AlmostEqual(f1, std::nexttoward(f1, +INFINITY), 1e-8f),
            "Wrong comparison with ULP-1 #f5");
  }

  {
    auto A = 1.0;
    for (auto i = 0; i < 6; ++i) {
      A += 1.0 / A - A / 2.0;
    }
    double sqrt_2 = A;
    double a      = 5.0 - sqrt_2;
    double b      = 5.0 + sqrt_2;
    auto   f1     = a * a / 23.0;
    auto   f2     = a / b;
    errorIf((not ulp_eq(f1, f2)) || (not cmp::AlmostEqual(f1, f2)),
            "Wrong comparison with ULP-1 #d1");
    errorIf(not cmp::AlmostZero((f1 - f2) * (f2 - f1)),
            "Wrong comparison with Zero #d2");
  }

  return 0;
}