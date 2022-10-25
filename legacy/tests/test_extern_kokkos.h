#ifndef TEST_EXTERN_KOKKOS_H
#define TEST_EXTERN_KOKKOS_H

#include "wrapper.h"
#include "timer.h"

#include <doctest.h>

#include <iostream>
#include <cstddef>
#include <cmath>

TEST_CASE("testing kokkos") {
  Kokkos::initialize();
  {
    REQUIRE(Kokkos::is_initialized());

    ntt::Timer timer("summation");

    int    N     = 10000000;
    double value = 16.695311, dvalue = 0.0001;
    auto   Sum = Lambda(const int i, double& sum) { sum += 1.0 / static_cast<double>(i + 1); };
    auto   Check = [&](const double sum) { return math::abs(value - sum) < dvalue; };

    double sum_var {0.0};
    timer.start();
    Kokkos::parallel_reduce("parallel_sum", N, Sum, sum_var);
    timer.stop();

    CHECK(Check(sum_var));
  }
  Kokkos::finalize();
  REQUIRE(!Kokkos::is_initialized());
}

#endif
