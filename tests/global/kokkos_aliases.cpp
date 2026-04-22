#include "arch/kokkos_aliases.h"

#include "global.h"

#include "utils/error.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <iostream>
#include <string>

auto main(int argc, char* argv[]) -> int {
  using namespace ntt;

  try {
    GlobalInitialize(argc, argv);

    {
      // simple arrays & ranges
      array_t<float*> a { "a", 100 };
      raise::ErrorIf(a.extent(0) != 100, "a.extent(0) must be 100", HERE);
      Kokkos::parallel_for(
        range_t<Dim::_1D>(0, 100),
        Lambda(const int i) { a(i) = static_cast<float>(i); });

      array_mirror_t<float*> b = Kokkos::create_mirror(a);
      Kokkos::deep_copy(b, a);

      Kokkos::parallel_for(range_h_t<Dim::_1D>(0, 100), [&](const int i) {
        raise::ErrorIf(b(i) != static_cast<float>(i),
                       "b(" + std::to_string(i) +
                         ") must be = " + std::to_string(i),
                       HERE);
      });
    }

    static_assert(std::is_same_v<ndarray_t<1>, array_t<real_t*>>);
    static_assert(std::is_same_v<ndarray_t<2>, array_t<real_t**>>);
    static_assert(std::is_same_v<ndarray_t<3>, array_t<real_t***>>);

    {
      // scatter arrays & ranges
      array_t<float*> a { "a", 100 };
      auto            a_scatter = Kokkos::Experimental::create_scatter_view(a);
      Kokkos::parallel_for(
        CreateRangePolicy<Dim::_1D>({ 0 }, { 100 }),
        Lambda(const int i) {
          auto a_acc     = a_scatter.access();
          a_acc(i % 10) += static_cast<float>(i);
        });
      Kokkos::Experimental::contribute(a, a_scatter);

      array_mirror_t<float*> b = Kokkos::create_mirror(a);
      Kokkos::deep_copy(b, a);
      Kokkos::parallel_for(
        CreateRangePolicyOnHost<Dim::_1D>({ 0 }, { 100 }),
        [&](const int i) {
          if (i < 10) {
            raise::ErrorIf(b(i) != (float)(450 + 10 * i),
                           "b(" + std::to_string(i) +
                             ") must be = " + std::to_string(450 + 10 * i),
                           HERE);
          } else {
            raise::ErrorIf(b(i) != 0.0, "b(" + std::to_string(i) + ") must be = 0.0", HERE);
          }
        });
    }

    static_assert(std::is_same_v<ndfield_t<Dim::_1D, 3>, array_t<real_t* [3]>>);
    static_assert(std::is_same_v<ndfield_t<Dim::_2D, 3>, array_t<real_t** [3]>>);
    static_assert(std::is_same_v<ndfield_t<Dim::_3D, 3>, array_t<real_t*** [3]>>);
    static_assert(std::is_same_v<ndfield_t<Dim::_1D, 6>, array_t<real_t* [6]>>);
    static_assert(std::is_same_v<ndfield_t<Dim::_2D, 6>, array_t<real_t** [6]>>);
    static_assert(std::is_same_v<ndfield_t<Dim::_3D, 6>, array_t<real_t*** [6]>>);
  }

  catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    GlobalFinalize();
    return 1;
  }

  GlobalFinalize();

  return 0;
}
