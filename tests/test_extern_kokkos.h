#ifndef TEST_EXTERN_KOKKOS_H
#define TEST_EXTERN_KOKKOS_H

#include "timer.h"
#include "math.h"

#ifdef KOKKOS
#include <Kokkos_Core.hpp>
#endif
#include "acutest/acutest.h"

#include <iostream>
#include <cmath>

#ifdef KOKKOS

void test_extern_kokkos(void) {
  {
    Kokkos::initialize();
    TEST_CHECK_ ( Kokkos::is_initialized(), "`Kokkos` initialize" );

    timer::Timer timer1("kokkos");
    timer::Timer timer2("serial");

    int N = 10000000;
    double value = 16.695311, dvalue = 0.0001;
    auto Sum = [=] (const int i, double & sum) {
      sum += 1.0 / static_cast<double>(i + 1); 
    };
    auto Check = [&] (const double sum) {
      return std::abs(value - sum) < dvalue;
    };

    double sum1 = 0.0;
    timer1.start();
    Kokkos::parallel_reduce(N, Sum, sum1); 
    timer1.stop();

    double sum2 = 0.0;
    timer2.start();
    for (int i { 0 }; i < N; ++i) {
      Sum(i, sum2);
    }
    timer2.stop();

    TEST_CHECK_ ( Check(sum1), "sum1 value is correct" );
    TEST_CHECK_ ( Check(sum2), "sum2 value is correct" );

    auto ms = timer::millisecond;
    TEST_CHECK_ ( timer1.getElapsedIn(ms) < timer2.getElapsedIn(ms), "Kokkos is faster" );

    Kokkos::finalize();
    TEST_CHECK_ ( true, "`Kokkos` finalize" );
  }
}
#else
void test_extern_kokkos(void) {
  TEST_CHECK_ ( true, "-- `Kokkos` is disabled, so the test is ignored" );
}
#endif

#endif
