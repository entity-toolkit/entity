#ifndef TEST_PIC_QSPHERICAL_H
#define TEST_PIC_QSPHERICAL_H

#include "wrapper.h"
#include "utils/qmath.h"

#include "pic.h"
#include "io/output.h"

#include <doctest.h>
#include <toml/toml.hpp>

#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>

TEST_CASE("testing PIC") {
  Kokkos::initialize();
  /* -------------------------------------------------------------------------- */
  /*                           Qspherical metric test                           */
  /* -------------------------------------------------------------------------- */
  SUBCASE("Qspherical") {}
  Kokkos::finalize();
}

#endif