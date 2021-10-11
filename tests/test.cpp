// test external libraries
#include "test_extern_toml.h"
#include "test_extern_kokkos.h"

#include <acutest.h>

void testSuccess() {}

TEST_LIST = {{"extern/toml", testExternToml},
             {"extern/kokkos", testExternKokkos},
             {"success", testSuccess},
             {nullptr, nullptr}};
