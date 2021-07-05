#ifndef TEST_LIB_AUX_H
#define TEST_LIB_AUX_H

#include "global.h"
#include "constants.h"
#include "mathematics.h"
#include "arrays.h"
#include "timer.h"

#include <acutest/acutest.h>

#include <cstddef>
#include <cmath>
#include <chrono>
#include <thread>

#include <iostream>

void testLibAux(void) {
  using namespace ntt::math;
  using namespace ntt::arrays;

  // `math`
  {
    TEST_CHECK_(true, "-- `real_t` uses %d bytes",
                static_cast<int>(sizeof(ntt::real_t)));
    // `double` comparison"
    TEST_CHECK(numbersAreEqual(
        0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1, 1.0));
    TEST_CHECK(numbersAreEqual(std::sin(ntt::constants::PI), 0.0));
    TEST_CHECK(!numbersAreEqual(1e14 + 0.1, 1e14));

    // `float` comparison"
    TEST_CHECK(numbersAreEqual(0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f +
                                   0.1f + 0.1f + 0.1f,
                               1.0f));
    TEST_CHECK(numbersAreEqual(static_cast<float>(std::sin(ntt::constants::PI)),
                               0.0f));
    TEST_CHECK(!numbersAreEqual(1e6f + 0.1f, 1e6f));
  }

  // `arrays`
  {
    // 1d arrays
    OneDArray<double> my1d(25);

    double x1 = 1.0;
    for (std::size_t i{0}; i < my1d.get_size(1); ++i) {
      double x2 = (4.0 / (8 * i + 1) - 2.0 / (8 * i + 4) - 1.0 / (8 * i + 5) -
                   1.0 / (8 * i + 6));
      my1d.set(i, x1 * x2);
      x1 /= 16.0;
    }
    double sum = 0.0;
    for (std::size_t i{0}; i < my1d.get_size(1); ++i) {
      sum += my1d.get(i);
    }
    TEST_CHECK(numbersAreEqual(sum, ntt::constants::PI));

    my1d.fillWith(4.0, true);
    sum = 0.0;
    for (std::size_t i{0}; i < my1d.get_size(1) + 2 * ntt::N_GHOSTS; ++i) {
      sum += my1d.get(i);
    }
    TEST_CHECK(numbersAreEqual(sum, 100.0 + (2 * ntt::N_GHOSTS) * 4.0));
    TEST_CHECK(my1d.getSizeInBytes() == 232);
  }

  // `timer`
  {
    TEST_CHECK_(true, "-- Using `%s` for timekeeping", ntt::timer::BACKEND);
    long double res;
    ntt::timer::Time testTime1(2.5, ntt::timer::millisecond);
    ntt::timer::Time testTime2(150.0, ntt::timer::microsecond);
    res = testTime1.getValue();
    TEST_CHECK(numbersAreEqual(res, 2.5));
    testTime1 = testTime2 + testTime1;
    res = testTime1.getValue();
    TEST_CHECK(numbersAreEqual(res, 2650.0));
    res = testTime1.represent(ntt::timer::second).getValue();
    TEST_CHECK(numbersAreEqual(res, 0.00265));

    // long double res;
    ntt::timer::Timer testTimer("test");
    testTimer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    testTimer.stop();
    res = testTimer.getElapsedIn(ntt::timer::second);
    TEST_CHECK(std::abs(res - 0.1) < 1e-1);
    res = testTimer.getElapsedIn(ntt::timer::nanosecond);
    TEST_CHECK(std::abs(res - 1e8) < 1e-1 * 1e8);
    TEST_MSG("t_ellapsed = %Lf", res);
    TEST_MSG("dt = %Lf", res - 1e8);
  }
}

#endif
