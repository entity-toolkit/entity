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
    const auto resolution = std::vector<unsigned int>({ 2500, 4000 });

    using namespace toml::literals::toml_literals;
#ifdef MINKOWSKI_METRIC
    const auto inputdata = R"(
      [domain]
      resolution  = [2500, 4000]
      extent      = [-50.0, 50.0, -20.0, 140.0]
      boundaries  = [["PERIODIC"], ["PERIODIC"]]

      [units]
      ppc0       = 1.0
      larmor0    = 1.0
      skindepth0 = 1.0
    )"_toml;
#else
    const auto inputdata = R"(
      [domain]
      resolution  = [2500, 4000]
      extent      = [1.0, 150.0]
      boundaries  = [["OPEN", "ABSORB"], ["AXIS"]]
      qsph_r0     = 0.0
      qsph_h      = 0.1
      a           = 0.95

      [units]
      ppc0       = 1.0
      larmor0    = 1.0
      skindepth0 = 1.0
    )"_toml;
#endif

    // ntt::SANDBOX<ntt::Dim2> sim(inputdata);
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();
    return -1;
  }
  ntt::GlobalFinalize();

  return 0;
}