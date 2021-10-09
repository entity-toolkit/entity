#ifndef TEST_EXTERN_KOKKOS_H
#define TEST_EXTERN_KOKKOS_H

#include "global.h"
#include "timer.h"

#include <acutest/acutest.h>

#include <iostream>
#include <cstddef>
#include <cmath>

void testExternKokkos(void) {
  Kokkos::initialize();
  {
    TEST_CHECK_(Kokkos::is_initialized(), "`Kokkos` initialize");

    ntt::Timer timer("summation");

    int N = 10000000;
    double value = 16.695311, dvalue = 0.0001;
    auto Sum = Lambda(const int i, double& sum) { sum += 1.0 / static_cast<double>(i + 1); };
    auto Check = [&](const double sum) { return std::abs(value - sum) < dvalue; };

    double sum_var {0.0};
    timer.start();
    Kokkos::parallel_reduce("parallel_sum", N, Sum, sum_var);
    timer.stop();

    TEST_CHECK_(Check(sum_var), "sum value is correct");

    TEST_CHECK_(true, std::to_string(timer.getElapsedIn(ntt::millisecond)).c_str());

    TEST_CHECK_(true, "`Kokkos` finalize");
  }
  Kokkos::finalize();
}

#endif
