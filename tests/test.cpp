// test local libraries
#include "test_lib_aux.h"

// test external libraries
#include "test_extern_toml.h"
#include "test_extern_kokkos.h"

#include <acutest/acutest.h>

void testSuccess(void) {
}

TEST_LIST = {
  { "lib/aux", testLibAux },
  { "extern/toml", testExternToml },
  { "extern/kokkos", testExternKokkos },
  { "success", testSuccess },
  { NULL, NULL }
};
