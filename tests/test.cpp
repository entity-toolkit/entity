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
auto main(int argc, char* argv[]) -> int {
  doctest::Context context;
  int              res;

  Kokkos::initialize();
  {
    plog::init(plog::info, "test.log");

#ifdef GPUENABLED
    throw std::runtime_error("tests should be done on CPUs");
#endif

    context.setOption("order-by", "none");

    context.applyCommandLine(argc, argv);

    context.setOption("no-intro", true);
    context.setOption("no-version", true);

    res = context.run();

    if (context.shouldExit()) { return res; }
  }
  Kokkos::finalize();
  return res;
}