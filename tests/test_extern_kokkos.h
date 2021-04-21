#ifndef TEST_EXTERN_KOKKOS_H
#define TEST_EXTERN_KOKKOS_H

#ifdef KOKKOS
#include <Kokkos_Core.hpp>
#endif
#include "acutest/acutest.h"

void test_extern_kokkos(void) {
  #ifdef KOKKOS
    Kokkos::initialize(argc, argv);
    {
      int N = (argc > 1) ? std::stoi(argv[1]) : 10000;
      int M = (argc > 2) ? std::stoi(argv[2]) : 10000;
      int R = (argc > 3) ? std::stoi(argv[3]) : 10;

      printf("Called with: %i %i %i\n", N, M, R);
      TEST_CHECK ( N == 10000 );
      TEST_CHECK ( M == 10000 );
      TEST_CHECK ( R == 10 );
    }
    Kokkos::finalize();
  #else
    TEST_CHECK_( true, "`Kokkos` is disabled, so the test is ignored");
  #endif
}

#endif
