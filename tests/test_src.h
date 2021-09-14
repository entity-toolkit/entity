#ifndef TEST_SRC_H
#define TEST_SRC_H

#include "global.h"
#include "constants.h"
#include "mathematics.h"
#include "domain.h"
#include "arrays.h"
#include "fields.h"
#include "timer.h"

#include <acutest/acutest.h>

#include <cstddef>
#include <cmath>
#include <chrono>
#include <thread>

void testSrc(void) {
  using namespace ntt::math;

  // `math`
  {
    TEST_CHECK_(true, "-- `real_t` uses %d bytes", static_cast<int>(sizeof(ntt::real_t)));
    // `double` comparison"
    TEST_CHECK(numbersAreEqual(0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1, 1.0));
    TEST_CHECK(numbersAreEqual(std::sin(ntt::constants::PI), 0.0));
    TEST_CHECK(!numbersAreEqual(1e14 + 0.1, 1e14));

    // `float` comparison"
    TEST_CHECK(numbersAreEqual(0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f + 0.1f, 1.0f));
    TEST_CHECK(numbersAreEqual(static_cast<float>(std::sin(ntt::constants::PI)), 0.0f));
    TEST_CHECK(!numbersAreEqual(1e6f + 0.1f, 1e6f));
  }

  // `arrays`
  {
    using namespace ntt::arrays;
    // 1d arrays
    OneDArray<double> my1d(25);

    double x1 {1.0};
    for (std::size_t i{0}; i < my1d.get_size(1); ++i) {
      double x2 = (4.0 / (8 * i + 1) - 2.0 / (8 * i + 4) - 1.0 / (8 * i + 5) - 1.0 / (8 * i + 6));
      my1d.set(i, x1 * x2);
      x1 /= 16.0;
    }
    double sum {0.0};
    for (std::size_t i{0}; i < my1d.get_size(1); ++i) {
      sum += my1d.get(i);
    }
    TEST_CHECK(numbersAreEqual(sum, ntt::constants::PI));

    my1d.fillWith(4.0);
    sum = 0.0;
    for (std::size_t i{0}; i < my1d.get_size(1); ++i) {
      sum += my1d.get(i);
    }
    TEST_CHECK(numbersAreEqual(sum, 100.0));
    TEST_CHECK(my1d.getSizeInBytes() == 200);
  }

  // `fields`
  {
    using namespace ntt::fields;
    // 1d fields
    OneDField<double> my1d(25);

    double x1 {1.0};
    for (std::size_t i{0}; i < my1d.get_size(1); ++i) {
      double x2 = (4.0 / (8 * i + 1) - 2.0 / (8 * i + 4) - 1.0 / (8 * i + 5) -
                   1.0 / (8 * i + 6));
      my1d.set(i, x1 * x2);
      x1 /= 16.0;
    }
    double sum {0.0};
    for (std::size_t i{0}; i < my1d.get_size(1); ++i) {
      sum += my1d.get(i);
    }
    TEST_CHECK(numbersAreEqual(sum, ntt::constants::PI));

    my1d.fillWith(4.0);
    sum = 0.0;
    for (std::size_t i{0}; i < my1d.get_size(1); ++i) {
      sum += my1d.get(i);
    }
    TEST_CHECK(numbersAreEqual(sum, 100.0 + (2 * ntt::N_GHOSTS) * 4.0));
    TEST_CHECK(my1d.getSizeInBytes() == 232);
  }

  {
    using namespace ntt::fields;
    // 3d fields
    ThreeDField<int> my3d(4, 11, 20);

    int N_GH {static_cast<int>(ntt::N_GHOSTS)};

    my3d.fillWith(1);
    int sum {0};
    // traversing the whole domain
    for (std::size_t i3{0}; i3 < my3d.get_size(3); ++i3) {
      for (std::size_t i2{0}; i2 < my3d.get_size(2); ++i2) {
        for (std::size_t i1{0}; i1 < my3d.get_size(1); ++i1) {
          sum += my3d.get(i1, i2, i3);
        }
      }
    }
    TEST_CHECK(sum == (4 + 2 * N_GH) * (11 + 2 * N_GH) * (20 + 2 * N_GH));
    my3d.fillWith(-1);
    // traversing only real domain (without ghosts)
    for (int i3{0}; i3 < my3d.get_extent(3); ++i3) {
      for (int i2{0}; i2 < my3d.get_extent(2); ++i2) {
        for (int i1{0}; i1 < my3d.get_extent(1); ++i1) {
          my3d.setAt(i1, i2, i3, 1);
        }
      }
    }
    sum = 0;
    // traversing only real domain (without ghosts)
    for (int i3{0}; i3 < my3d.get_extent(3); ++i3) {
      for (int i2{0}; i2 < my3d.get_extent(2); ++i2) {
        for (int i1{0}; i1 < my3d.get_extent(1); ++i1) {
          sum += my3d.getAt(i1, i2, i3);
        }
      }
    }
    TEST_CHECK(sum == 4 * 11 * 20);
    sum = 0;
    // traversing the whole domain
    for (std::size_t i3{0}; i3 < my3d.get_size(3); ++i3) {
      for (std::size_t i2{0}; i2 < my3d.get_size(2); ++i2) {
        for (std::size_t i1{0}; i1 < my3d.get_size(1); ++i1) {
          sum += my3d.get(i1, i2, i3);
        }
      }
    }
    int result {4 * 11 * 20 - 11 * 4 * N_GH * 2 - 4 * 20 * N_GH * 2 - 20 * 11 * N_GH * 2 - 8 * N_GH * N_GH * N_GH - 11 * N_GH * N_GH * 4 - 4 * N_GH * N_GH * 4 - 20 * N_GH * N_GH * 4};
    TEST_CHECK(sum == result);
    sum = 0;
    // traversing the whole domain again
    for (int i3{-N_GH}; i3 < my3d.get_extent(3) + N_GH; ++i3) {
      for (int i2{-N_GH}; i2 < my3d.get_extent(2) + N_GH; ++i2) {
        for (int i1{-N_GH}; i1 < my3d.get_extent(1) + N_GH; ++i1) {
          sum += my3d.getAt(i1, i2, i3);
        }
      }
    }
    TEST_CHECK(sum == result);
  }

  // `domain`
  {
    using namespace ntt;
    Domain my_domain(TWO_D, POLAR_COORD);
    my_domain.set_extent({0.1, 2.0, 0.0, 2.0 * constants::PI});
    my_domain.set_resolution({100, 25});
    my_domain.set_boundaries({OPEN_BC, PERIODIC_BC});

    auto dxi = my_domain.dxi();
    TEST_CHECK(numbersAreEqual(dxi[0], 0.019) && numbersAreEqual(dxi[1], 2.0 * constants::PI / 25));
    auto sizexi = my_domain.sizexi();
    TEST_CHECK(numbersAreEqual(sizexi[0], 1.9) && numbersAreEqual(sizexi[1], 2.0 * constants::PI));

    auto ijk = my_domain.x1x2x3_to_ijk({1.5, 1.2});
    auto x1x2x3 = my_domain.ijk_to_x1x2x3({66, 15});
    TEST_CHECK((ijk[0] == 73) && (ijk[1] == 4));
    TEST_CHECK(numbersAreEqual(x1x2x3[0], 1.354) && numbersAreEqual(x1x2x3[1], 3.769911184307752));
    ijk = my_domain.x1x2x3_to_ijk(x1x2x3);
    TEST_CHECK((ijk[0] == 66) && (ijk[1] == 15));
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
