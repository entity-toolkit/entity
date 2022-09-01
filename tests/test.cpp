#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

#ifdef MINKOWSKI_METRIC

#  include "pic/test_pic_minkowski.h"

#elif defined(SPHERICAL_METRIC)

//

#elif defined(QSPHERICAL_METRIC)

#  include "pic/test_pic_qsph.h"

#elif defined(KERR_SCHILD_METRIC)

#elif defined(QKERR_SCHILD_METRIC)

#else

// test external libraries
#  include "test_extern_kokkos.h"
#  include "test_extern_toml.h"

// test utils
#  include "test_qmath.h"

#endif

#include <plog/Log.h>
#include <plog/Initializers/RollingFileInitializer.h>

#include <stdexcept>

// Logging is done via `plog` library...
// ... Use the following commands:
//  `PLOGI << ...` for general info
//  `PLOGF << ...` for fatal error messages (development)
//  `PLOGD << ...` for debug messages (development)
//  `PLOGE << ...` for simple error messages
//  `PLOGW << ...` for warnings

auto main(int argc, char* argv[]) -> int {
  plog::init(plog::info, "test.log");

#ifdef GPUENABLED
  throw std::runtime_error("tests should be done on CPUs");
#endif

  doctest::Context context;
  context.setOption("order-by", "none");

  context.applyCommandLine(argc, argv);

  context.setOption("no-intro", true);
  context.setOption("no-version", true);

  int res = context.run();

  if (context.shouldExit()) { return res; }

  return res;
}