#include "archetypes/spatial_dist.h"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"

#include "metrics/minkowski.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>

using namespace ntt;
using namespace metric;
using namespace arch;

template <class SpDist, class M>
struct Caller {
  static_assert(M::CoordType == Coord::Cart,
                "Caller only available in Cartesian coordinates");
  static_assert(M::Dim != Dim::_1D, "1D Caller not available");

  Caller(const SpDist& dist, const M& metric)
    : dist { dist }
    , metric { metric } {}

  Inline void operator()(index_t i1, index_t i2) const {
    if constexpr (M::Dim == Dim::_2D) {
      coord_t<M::Dim> x_Code { static_cast<real_t>(i1), static_cast<real_t>(i2) };
      coord_t<M::Dim> x_Sph { ZERO };
      metric.template convert<Crd::Cd, Crd::Sph>(x_Code, x_Sph);
      const auto D        = dist(x_Code);
      const auto D_expect = math::sqrt(SQR(x_Sph[0]) + SQR(x_Sph[1]));
      if (not cmp::AlmostEqual(D, D_expect)) {
        raise::KernelError(HERE, "incorrect radial dist");
      }
    } else {
      raise::KernelError(HERE, "2D called for non-2D");
    }
  }

  Inline void operator()(index_t i1, index_t i2, index_t i3) const {
    if constexpr (M::Dim == Dim::_3D) {
      coord_t<M::Dim> x_Code { static_cast<real_t>(i1),
                               static_cast<real_t>(i2),
                               static_cast<real_t>(i3) };
      coord_t<M::Dim> x_Sph { ZERO };
      metric.template convert<Crd::Cd, Crd::Sph>(x_Code, x_Sph);
      const auto D        = dist(x_Code);
      const auto D_expect = math::sqrt(
        SQR(x_Sph[0]) + SQR(x_Sph[1]) + SQR(x_Sph[2]));
      if (not cmp::AlmostEqual(D, D_expect)) {
        raise::KernelError(HERE, "incorrect radial dist");
      }
    } else {
      raise::KernelError(HERE, "3D called for non-3D");
    }
  }

private:
  SpDist  dist;
  const M metric;
};

template <SimEngine::type S, class M>
struct RadialDist : public SpatialDistribution<S, M> {
  using SpatialDistribution<S, M>::metric;
  static_assert(M::CoordType == Coord::Cart,
                "RadialDist only available in Cartesian coordinates");

  static_assert(M::Dim != Dim::_1D, "1D RadialDist not available");

  RadialDist(const M& metric) : SpatialDistribution<S, M> { metric } {}

  auto operator()(const coord_t<M::Dim>& x_Code) const -> real_t {
    coord_t<M::Dim> x_Sph { ZERO };
    metric.template convert<Crd::Cd, Crd::Sph>(x_Code, x_Sph);
    auto r { ZERO };
    for (dim_t d { 0u }; d < M::Dim; ++d) {
      r += SQR(x_Sph[d]);
    }
    return math::sqrt(r);
  }
};

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    Minkowski<Dim::_2D> m1 {
      {              10,              10 },
      { { -10.0, 55.0 }, { -10.0, 55.0 } }
    };
    RadialDist<SimEngine::SRPIC, Minkowski<Dim::_2D>> r1 { m1 };

    Minkowski<Dim::_3D> m2 {
      {            10,            10,            30 },
      { { -1.0, 1.0 }, { -1.0, 1.0 }, { -3.0, 3.0 } }
    };
    RadialDist<SimEngine::SRPIC, Minkowski<Dim::_3D>> r2 { m2 };

    Kokkos::parallel_for(
      "RadialDist 2D",
      CreateRangePolicy<Dim::_2D>({ 0, 0 }, { 10, 10 }),
      Caller<RadialDist<SimEngine::SRPIC, Minkowski<Dim::_2D>>, Minkowski<Dim::_2D>>(
        r1,
        m1));

    Kokkos::parallel_for(
      "RadialDist 3D",
      CreateRangePolicy<Dim::_3D>({ 0, 0, 0 }, { 10, 10, 30 }),
      Caller<RadialDist<SimEngine::SRPIC, Minkowski<Dim::_3D>>, Minkowski<Dim::_3D>>(
        r2,
        m2));

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
