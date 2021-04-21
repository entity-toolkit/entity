// test local libraries
#include "test_lib_aux.h"

// test external libraries
#include "test_extern_toml.h"

#include "acutest/acutest.h"

void test_success(void) {
}

TEST_LIST = {
    { "lib/aux", test_lib_aux },
    { "extern/toml", test_extern_toml },
    { "success", test_success },
    { NULL, NULL }
};
