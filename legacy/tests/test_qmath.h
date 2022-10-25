#ifndef TEST_QMATH_H
#define TEST_QMATH_H

#include "qmath.h"

#include <doctest.h>

#include <vector>

TEST_CASE("testing qmath AlmostEqual") {
  SUBCASE("for floats") {
    SUBCASE("equal") {
      std::vector<float> a {1000000.0f,
                            -1000000.0f,
                            1.0000001f,
                            -1.0000001f,
                            0.000000001000001f,
                            -0.000000001000001f,
                            0.3f,
                            -0.3f,
                            0.0f,
                            -0.0f};
      std::vector<float> b {1000001.0f,
                            -1000001.0f,
                            1.00000002f,
                            -1.00000002f,
                            0.000000001000002f,
                            -0.000000001000002f,
                            0.30000003f,
                            -0.30000003f,
                            0.0f,
                            0.0f};
      for (std::size_t i {0}; i < a.size(); ++i) {
        CHECK_MESSAGE(ntt::AlmostEqual(a[i], b[i]),
                      "a[" << i << "] = " << a[i] << " != b[" << i << "] = " << b[i]
                           << " want: ==");
        CHECK_MESSAGE(ntt::AlmostEqual(b[i], a[i]),
                      "b[" << i << "] = " << b[i] << " != a[" << i << "] = " << a[i]
                           << " want: ==");
      }
    }
    SUBCASE("not equal") {
      std::vector<float> a {10000.0f,
                            -10000.0f,
                            1.0002f,
                            -1.0002f,
                            0.000000000001002f,
                            -0.000000000001002f,
                            0.00000001f,
                            0.00000001f,
                            1.000000001f,
                            -1.000000001f};
      std::vector<float> b {10001.0f,
                            -10001.0f,
                            1.0001f,
                            -1.0001f,
                            0.000000000001001f,
                            -0.000000000001001f,
                            0.0f,
                            0.0f,
                            -1.0f,
                            1.0f};
      for (std::size_t i {0}; i < a.size(); ++i) {
        CHECK_MESSAGE(!ntt::AlmostEqual(a[i], b[i]),
                      "a[" << i << "] = " << a[i] << " == b[" << i << "] = " << b[i]
                           << " want: !=");
        CHECK_MESSAGE(!ntt::AlmostEqual(b[i], a[i]),
                      "b[" << i << "] = " << b[i] << " == a[" << i << "] = " << a[i]
                           << " want: !=");
      }
    }
  }

  SUBCASE("for doubles") {
    SUBCASE("equal") {
      std::vector<double> a {10000000000000000.0, 1e-12, 1e-13, 0.0, 1e-12, -1e-13};
      std::vector<double> b {10000000000000001.0, 1e-13, -1e-14, 0.0, 0.0, 0.0};
      for (std::size_t i {0}; i < a.size(); ++i) {
        CHECK_MESSAGE(ntt::AlmostEqual(a[i], b[i]),
                      "a[" << i << "] = " << a[i] << " != b[" << i << "] = " << b[i]
                           << " want: ==");
        CHECK_MESSAGE(ntt::AlmostEqual(b[i], a[i]),
                      "b[" << i << "] = " << b[i] << " != a[" << i << "] = " << a[i]
                           << " want: ==");
      }
    }
    SUBCASE("not equal") {
      std::vector<double> a {1000000000000000.0, 1e-11, 1e-12, 1e-11, -1e-11};
      std::vector<double> b {1000000000000001.0, 1e-12, -1e-13, 0.0, 0.0};
      for (std::size_t i {0}; i < a.size(); ++i) {
        CHECK_MESSAGE(!ntt::AlmostEqual(a[i], b[i]),
                      "a[" << i << "] = " << a[i] << " == b[" << i << "] = " << b[i]
                           << " want: !=");
        CHECK_MESSAGE(!ntt::AlmostEqual(b[i], a[i]),
                      "b[" << i << "] = " << b[i] << " == a[" << i << "] = " << a[i]
                           << " want: !=");
      }
    }
  }
}

#endif
