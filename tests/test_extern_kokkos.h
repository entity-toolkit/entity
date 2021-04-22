#ifndef TEST_EXTERN_KOKKOS_H
#define TEST_EXTERN_KOKKOS_H

#ifdef KOKKOS
#include <Kokkos_Core.hpp>
#endif
#include "acutest/acutest.h"

#ifdef KOKKOS
void test_extern_kokkos(void) {
  {
    Kokkos::initialize();
    TEST_CHECK_ ( true, "`Kokkos` initialize done" );
    Kokkos::finalize();
    TEST_CHECK_ ( true, "`Kokkos` finalize done" );
  }
}
#else
void test_extern_kokkos(void) {
  TEST_CHECK_( true, "-- `Kokkos` is disabled, so the test is ignored");
}
#endif

#endif
