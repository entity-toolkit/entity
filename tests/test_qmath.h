#ifndef TEST_QMATH_H
#define TEST_QMATH_H

#include "qmath.h"

void testQmath(void) {

  // Floats
  TEST_CHECK(ntt::AlmostEqual(1000000.0f, 1000001.0f));
  TEST_CHECK(ntt::AlmostEqual(1000001.0f, 1000000.0f));
  TEST_CHECK(!ntt::AlmostEqual(10000.0f, 10001.0f));
  TEST_CHECK(!ntt::AlmostEqual(10001.0f, 10000.0f));

  TEST_CHECK(ntt::AlmostEqual(-1000000.0f, -1000001.0f));
  TEST_CHECK(ntt::AlmostEqual(-1000001.0f, -1000000.0f));
  TEST_CHECK(!ntt::AlmostEqual(-10000.0f, -10001.0f));
  TEST_CHECK(!ntt::AlmostEqual(-10001.0f, -10000.0f));

  TEST_CHECK(ntt::AlmostEqual(1.0000001f, 1.0000002f));
  TEST_CHECK(ntt::AlmostEqual(1.0000002f, 1.0000001f));
  TEST_CHECK(!ntt::AlmostEqual(1.0002f, 1.0001f));
  TEST_CHECK(!ntt::AlmostEqual(1.0001f, 1.0002f));

  TEST_CHECK(ntt::AlmostEqual(-1.000001f, -1.000002f));
  TEST_CHECK(ntt::AlmostEqual(-1.000002f, -1.000001f));
  TEST_CHECK(!ntt::AlmostEqual(-1.0001f, -1.0002f));
  TEST_CHECK(!ntt::AlmostEqual(-1.0002f, -1.0001f));

  TEST_CHECK(ntt::AlmostEqual(0.000000001000001f, 0.000000001000002f));
  TEST_CHECK(ntt::AlmostEqual(0.000000001000002f, 0.000000001000001f));
  TEST_CHECK(!ntt::AlmostEqual(0.000000000001002f, 0.000000000001001f));
  TEST_CHECK(!ntt::AlmostEqual(0.000000000001001f, 0.000000000001002f));

  TEST_CHECK(ntt::AlmostEqual(-0.000000001000001f, -0.000000001000002f));
  TEST_CHECK(ntt::AlmostEqual(-0.000000001000002f, -0.000000001000001f));
  TEST_CHECK(!ntt::AlmostEqual(-0.000000000001002f, -0.000000000001001f));
  TEST_CHECK(!ntt::AlmostEqual(-0.000000000001001f, -0.000000000001002f));

  TEST_CHECK(ntt::AlmostEqual(0.3f, 0.30000003f));
  TEST_CHECK(ntt::AlmostEqual(-0.3f, -0.30000003f));

  TEST_CHECK(ntt::AlmostEqual(0.0f, 0.0f));
  TEST_CHECK(ntt::AlmostEqual(0.0f, -0.0f));
  TEST_CHECK(ntt::AlmostEqual(-0.0f, -0.0f));
  TEST_CHECK(!ntt::AlmostEqual(0.00000001f, 0.0f));
  TEST_CHECK(!ntt::AlmostEqual(0.0f, 0.00000001f));
  TEST_CHECK(!ntt::AlmostEqual(-0.00000001f, 0.0f));
  TEST_CHECK(!ntt::AlmostEqual(0.0f, -0.00000001f));

  TEST_CHECK(ntt::AlmostEqual(0.0f, 1e-40f, 0.01f));
  TEST_CHECK(ntt::AlmostEqual(1e-40f, 0.0f, 0.01f));
  TEST_CHECK(!ntt::AlmostEqual(1e-40f, 0.0f, 0.000001f));
  TEST_CHECK(!ntt::AlmostEqual(0.0f, 1e-40f, 0.000001f));

  TEST_CHECK(ntt::AlmostEqual(0.0f, -1e-40f, 0.1f));
  TEST_CHECK(ntt::AlmostEqual(-1e-40f, 0.0f, 0.1f));
  TEST_CHECK(!ntt::AlmostEqual(-1e-40f, 0.0f, 0.00000001f));
  TEST_CHECK(!ntt::AlmostEqual(0.0f, -1e-40f, 0.00000001f));

  TEST_CHECK(!ntt::AlmostEqual(1.000000001f, -1.0f));
  TEST_CHECK(!ntt::AlmostEqual(-1.0f, 1.000000001f));
  TEST_CHECK(!ntt::AlmostEqual(-1.000000001f, 1.0f));
  TEST_CHECK(!ntt::AlmostEqual(1.0f, -1.000000001f));

  // Doubles
  TEST_CHECK(ntt::AlmostEqual(10000000.0, 10000001.0));
  TEST_CHECK(ntt::AlmostEqual(100000000.0, 100000001.0));
  TEST_CHECK(ntt::AlmostEqual(1000000000.0, 1000000001.0));
  TEST_CHECK(ntt::AlmostEqual(10000000000.0, 10000000001.0));
  TEST_CHECK(ntt::AlmostEqual(100000000000.0, 100000000001.0));
  TEST_CHECK(ntt::AlmostEqual(1000000000000.0, 1000000000001.0));
  // TEST_CHECK(ntt::AlmostEqual(100000000001.0, 100000000000.0));
  // TEST_CHECK(!ntt::AlmostEqual(10000.0f, 10001.0f));
  // TEST_CHECK(!ntt::AlmostEqual(10001.0f, 10000.0f));

  // TEST_CHECK(ntt::AlmostEqual(-1000000.0f, -1000001.0f));
  // TEST_CHECK(ntt::AlmostEqual(-1000001.0f, -1000000.0f));
  // TEST_CHECK(!ntt::AlmostEqual(-10000.0f, -10001.0f));
  // TEST_CHECK(!ntt::AlmostEqual(-10001.0f, -10000.0f));

  // TEST_CHECK(ntt::AlmostEqual(1.0000001f, 1.0000002f));
  // TEST_CHECK(ntt::AlmostEqual(1.0000002f, 1.0000001f));
  // TEST_CHECK(!ntt::AlmostEqual(1.0002f, 1.0001f));
  // TEST_CHECK(!ntt::AlmostEqual(1.0001f, 1.0002f));

  // TEST_CHECK(ntt::AlmostEqual(-1.000001f, -1.000002f));
  // TEST_CHECK(ntt::AlmostEqual(-1.000002f, -1.000001f));
  // TEST_CHECK(!ntt::AlmostEqual(-1.0001f, -1.0002f));
  // TEST_CHECK(!ntt::AlmostEqual(-1.0002f, -1.0001f));

  // TEST_CHECK(ntt::AlmostEqual(0.000000001000001f, 0.000000001000002f));
  // TEST_CHECK(ntt::AlmostEqual(0.000000001000002f, 0.000000001000001f));
  // TEST_CHECK(!ntt::AlmostEqual(0.000000000001002f, 0.000000000001001f));
  // TEST_CHECK(!ntt::AlmostEqual(0.000000000001001f, 0.000000000001002f));

  // TEST_CHECK(ntt::AlmostEqual(-0.000000001000001f, -0.000000001000002f));
  // TEST_CHECK(ntt::AlmostEqual(-0.000000001000002f, -0.000000001000001f));
  // TEST_CHECK(!ntt::AlmostEqual(-0.000000000001002f, -0.000000000001001f));
  // TEST_CHECK(!ntt::AlmostEqual(-0.000000000001001f, -0.000000000001002f));

  // TEST_CHECK(ntt::AlmostEqual(0.3f, 0.30000003f));
  // TEST_CHECK(ntt::AlmostEqual(-0.3f, -0.30000003f));

  // TEST_CHECK(ntt::AlmostEqual(0.0f, 0.0f));
  // TEST_CHECK(ntt::AlmostEqual(0.0f, -0.0f));
  // TEST_CHECK(ntt::AlmostEqual(-0.0f, -0.0f));
  // TEST_CHECK(!ntt::AlmostEqual(0.00000001f, 0.0f));
  // TEST_CHECK(!ntt::AlmostEqual(0.0f, 0.00000001f));
  // TEST_CHECK(!ntt::AlmostEqual(-0.00000001f, 0.0f));
  // TEST_CHECK(!ntt::AlmostEqual(0.0f, -0.00000001f));

  // TEST_CHECK(ntt::AlmostEqual(0.0f, 1e-40f, 0.01f));
  // TEST_CHECK(ntt::AlmostEqual(1e-40f, 0.0f, 0.01f));
  // TEST_CHECK(!ntt::AlmostEqual(1e-40f, 0.0f, 0.000001f));
  // TEST_CHECK(!ntt::AlmostEqual(0.0f, 1e-40f, 0.000001f));

  // TEST_CHECK(ntt::AlmostEqual(0.0f, -1e-40f, 0.1f));
  // TEST_CHECK(ntt::AlmostEqual(-1e-40f, 0.0f, 0.1f));
  // TEST_CHECK(!ntt::AlmostEqual(-1e-40f, 0.0f, 0.00000001f));
  // TEST_CHECK(!ntt::AlmostEqual(0.0f, -1e-40f, 0.00000001f));

  // TEST_CHECK(!ntt::AlmostEqual(1.000000001f, -1.0f));
  // TEST_CHECK(!ntt::AlmostEqual(-1.0f, 1.000000001f));
  // TEST_CHECK(!ntt::AlmostEqual(-1.000000001f, 1.0f));
  // TEST_CHECK(!ntt::AlmostEqual(1.0f, -1.000000001f));
}

#endif