#ifndef TEST_EXTERN_KOKKOS_H
#define TEST_EXTERN_KOKKOS_H

#ifdef KOKKOS
#include <Kokkos_Core.hpp>
#endif
#include "acutest/acutest.h"

void test_extern_kokkos(void) {
  #ifdef KOKKOS
    Kokkos::initialize();
    Kokkos::finalize();
  #else
    TEST_CHECK_( true, "`Kokkos` is disabled, so the test is ignored");
  #endif
}

#endif
