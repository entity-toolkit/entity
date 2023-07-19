#include "wrapper.h"

#include "sandbox.h"

#include "communications/decomposition.h"
#include "communications/metadomain.h"
#include "utils/qmath.h"

#include <toml.hpp>

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    toml::table simulation, domain, units, output;
    simulation["title"]  = "WriterTest";
    domain["resolution"] = toml::array { 2500, 4000 };

#ifdef MINKOWSKI_METRIC
    domain["extent"] = toml::array { -50.0, 50.0, -20.0, 140.0 };
    domain["boundaries"]
      = toml::array { toml::array { "PERIODIC" }, toml::array { "PERIODIC" } };
#else
    domain["extent"]     = toml::array { 1.0, 150.0 };
    domain["boundaries"] = toml::array {
      toml::array { "OPEN", "ABSORB" },
       toml::array { "AXIS" }
    };
#endif

    units["ppc0"]       = 1.0;
    units["larmor0"]    = 1.0;
    units["skindepth0"] = 1.0;

    output["fields"]    = toml::array { "E", "B" };
    output["format"]    = "HDF5";

    auto inputdata      = toml::table {
           {"simulation", simulation},
           {    "domain",     domain},
           {     "units",      units},
           {    "output",     output}
    };

    ntt::SANDBOX<ntt::Dim2> sim(inputdata);
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();
    return -1;
  }
  ntt::GlobalFinalize();

  return 0;
}