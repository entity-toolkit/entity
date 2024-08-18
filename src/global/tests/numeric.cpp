#include "utils/numeric.h"

#include "global.h"

#include "utils/comparators.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

template <typename T>
void testVec() {
  const T a = 0.256 + sizeof(T) / 5.0;
  const T b = -0.512 + sizeof(T) / 5.0;
  const T c = 0.768 + sizeof(T) / 5.0;

  const T a1 = math::cos(a) * math::cos(c) -
               math::sin(a) * math::sin(b) * math::sin(c);
  const T a2 = math::cos(b) * math::sin(a);
  const T a3 = math::cos(c) * math::sin(a) * math::sin(b) +
               math::cos(a) * math::sin(c);

  const T b1 = -math::cos(b) * math::sin(c);
  const T b2 = -math::sin(b);
  const T b3 = math::cos(b) * math::cos(c);

  constexpr auto c_one = static_cast<T>(1.0);

  errorIf(not cmp::AlmostZero(DOT(a1, a2, a3, b1, b2, b3)),
          "DOT of perp vectors != 0");
  errorIf(not cmp::AlmostEqual(NORM_SQR(a1, a2, a3), c_one), "NORM_SQR of a != 1");
  errorIf(not cmp::AlmostEqual(NORM(a1, a2, a3), c_one), "NORM of a != 1");
  errorIf(not cmp::AlmostEqual(NORM_SQR(b1, b2, b3), c_one), "NORM_SQR of b != 1");
  errorIf(not cmp::AlmostEqual(NORM(b1, b2, b3), c_one), "NORM of b != 1");

  const T c1 = CROSS_x1(a1, a2, a3, b1, b2, b3);
  const T c2 = CROSS_x2(a1, a2, a3, b1, b2, b3);
  const T c3 = CROSS_x3(a1, a2, a3, b1, b2, b3);

  // decreasing accuracy
  constexpr auto eps = 10 * std::numeric_limits<T>::epsilon();
  errorIf(not cmp::AlmostEqual(NORM_SQR(c1, c2, c3), c_one, eps),
          "NORM_SQR of c != 1");
  errorIf(not cmp::AlmostEqual(NORM(c1, c2, c3), c_one, eps), "NORM of c != 1");

  errorIf(not cmp::AlmostZero(DOT(a1, a2, a3, c1, c2, c3), eps),
          "DOT of a and c != 0");
  errorIf(not cmp::AlmostZero(DOT(b1, b2, b3, c1, c2, c3), eps),
          "DOT of b and c != 0");
}

auto main() -> int {
  errorIf(IMIN(1, 2) != 1, "IMIN(1, 2) != 1");
  errorIf(IMIN(2, 1) != 1, "IMIN(2, 1) != 1");
  errorIf(IMAX(1, 2) != 2, "IMAX(1, 2) != 2");
  errorIf(IMAX(2, 1) != 2, "IMAX(2, 1) != 2");

  errorIf(SIGN(-1) != -ONE, "SIGN(-1) != -1");
  errorIf(SIGN(1) != ONE, "SIGN(1) != 1");

  errorIf(HEAVISIDE(-1) != ZERO, "HEAVISIDE(-1) != 0");
  errorIf(HEAVISIDE(0) != ZERO, "HEAVISIDE(0) != 0");
  errorIf(HEAVISIDE(1) != ONE, "HEAVISIDE(1) != 1");

  errorIf(SQR(2) != 4, "SQR(2) != 4");
  errorIf(CUBE(2) != 8, "CUBE(2) != 8");

  // dot product of perp 3D vectors
  testVec<float>();
  testVec<double>();
  return 0;
}