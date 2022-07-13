#ifndef TEST_PIC_MINKOWSKI_H
#define TEST_PIC_MINKOWSKI_H

#include "global.h"
#include "pic.h"
#include "qmath.h"

#include <toml/toml.hpp>
#include <acutest.h>

#include <string>
#include <iostream>

void testPICMinkowski2D(void) {
  // TEST_CHECK(ntt::AlmostEqual(1000000f, 1000001f));
  // TEST_CHECK(ntt::AlmostEqual(1000001f, 1000000f));
  // TEST_CHECK(!ntt::AlmostEqual(10000f, 10001f));
  // TEST_CHECK(!ntt::AlmostEqual(10001f, 10000f));

  // std::string                     input_toml = R"TOML(
  //   [domain]
  //   resolution      = [64, 64]
  //   extent          = [-2.0, 2.0, -2.0, 2.0]
  //   boundaries      = ["PERIODIC", "PERIODIC"]

  //   [units]
  //   ppc0            = 1.0
  //   larmor0         = 0.1
  //   skindepth0      = 0.1

  //   [particles]
  //   n_species       = 2

  //   [species_1]
  //   mass            = 1.0
  //   charge          = -1.0
  //   maxnpart        = 1e1

  //   [species_2]
  //   mass            = 1.0
  //   charge          = 1.0
  //   maxnpart        = 1e1
  //   )TOML";
  // std::istringstream              is(input_toml, std::ios_base::binary | std::ios_base::in);
  // auto                            inputdata = toml::parse(is, "std::string");
  // ntt::PIC<ntt::Dimension::TWO_D> sim(inputdata);
  // sim.initialize();
  // std::cout << sim.mblock().timestep() << "\n";
}

void testPICMinkowski(void) {
  Kokkos::initialize();
  { testPICMinkowski2D(); }
  Kokkos::finalize();
}

#endif