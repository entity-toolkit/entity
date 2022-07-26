#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

// test external libraries
#include "test_extern_kokkos.h"
#include "test_extern_toml.h"

// test auxiliary
#include "test_qmath.h"

// testing the core
#include "pic/test_pic_minkowski.h"

#include <stdexcept>

auto main(int argc, char** argv) -> int {
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
