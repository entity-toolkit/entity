#ifndef TEST_LIB_AUX_H
#define TEST_LIB_AUX_H

#include "constants.h"
#include "math.h"
#include "arrays.h"

#include "acutest/acutest.h"

#include <cstddef>
#include <cmath>

void test_lib_aux(void) {
  using namespace math;
  using namespace arrays;

  // `math`

  // `double` comparison"
  TEST_CHECK (numbersAreEqual(0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1, 1.0));
  TEST_CHECK (numbersAreEqual(std::sin(constants::PI), 0.0));
  TEST_CHECK (!numbersAreEqual(1e14 + 0.1, 1e14));

  // `float` comparison"
  TEST_CHECK (numbersAreEqual(0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f, 1.0f));
  TEST_CHECK (numbersAreEqual(static_cast<float>(std::sin(constants::PI)), 0.0f));
  TEST_CHECK (!numbersAreEqual(1e6f + 0.1f, 1e6f));

  // `arrays`

  // 1d arrays
  OneDArray<double> my1d(25);

  double x1 = 1.0;
  for (std::size_t i { 0 }; i < my1d.getDim(1); ++i) {
    double x2 = (4.0 / (8 * i + 1) - 2.0 / (8 * i + 4) - 1.0 / (8 * i + 5) - 1.0 / (8 * i + 6));
    my1d.set(i, x1 * x2);
    x1 /= 16.0;
  }
  double sum = 0.0;
  for (std::size_t i { 0 }; i < my1d.getDim(1); ++i) {
    sum += my1d.get(i);
  }
  TEST_CHECK (numbersAreEqual(sum, constants::PI));

  my1d.fillWith(4.0);
  sum = 0.0;
  for (std::size_t i { 0 }; i < my1d.getDim(1); ++i) {
    sum += my1d.get(i);
  }
  TEST_CHECK (numbersAreEqual(sum, 100.0));
}

#endif
