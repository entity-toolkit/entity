#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/formatting.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "kernels/particle_pusher_sr.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ntt;
using namespace metric;

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

template <typename T>
void put_value(array_t<T*>& arr, T v, index_t p) {
  auto h = Kokkos::create_mirror_view(arr);
  Kokkos::deep_copy(h, arr);
  h(p) = v;
  Kokkos::deep_copy(arr, h);
}

inline static constexpr auto epsilon = std::numeric_limits<real_t>::epsilon();

struct Pgen {
  static constexpr auto is_pgen { true };
};

template <SimEngine::type S, typename M>
void testGCAPusher(const std::vector<std::size_t>&      res,
                   const boundaries_t<real_t>&          ext,
                   const std::map<std::string, real_t>& params = {},
                   const real_t                         acc    = ONE) {
  static_assert(M::Dim == 3);
  errorIf(res.size() != M::Dim, "res.size() != M::Dim");

  boundaries_t<real_t> extent;
  if constexpr (M::CoordType == Coord::Cart) {
    extent = ext;
  } else {
    extent = {
      ext[0],
      {ZERO, constant::PI}
    };
  }

  M metric { res, extent, params };

  const int nx1 = res[0];
  const int nx2 = res[1];
  const int nx3 = res[2];

  auto coeff = real_t { 1.0 };
  auto dt    = real_t { 1.0 };

  const auto range_ext = CreateRangePolicy<Dim::_3D>(
    { 0, 0, 0 },
    { res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS });

  auto emfield = ndfield_t<Dim::_3D, 6> { "emfield",
                                          res[0] + 2 * N_GHOSTS,
                                          res[1] + 2 * N_GHOSTS,
                                          res[2] + 2 * N_GHOSTS };

  Kokkos::parallel_for(
    "init 3D",
    range_ext,
    Lambda(index_t i1, index_t i2, index_t i3) {
      emfield(i1, i2, i3, em::ex1) = 0.0;
      emfield(i1, i2, i3, em::ex2) = 0.0;
      emfield(i1, i2, i3, em::ex3) = 0.0;
      emfield(i1, i2, i3, em::bx1) = 0.22;
      emfield(i1, i2, i3, em::bx2) = 0.44;
      emfield(i1, i2, i3, em::bx3) = 0.66;
    });

  array_t<int*>      i1 { "i1", 2 };
  array_t<int*>      i2 { "i2", 2 };
  array_t<int*>      i3 { "i3", 2 };
  array_t<prtldx_t*> dx1 { "dx1", 2 };
  array_t<prtldx_t*> dx2 { "dx2", 2 };
  array_t<prtldx_t*> dx3 { "dx3", 2 };
  array_t<real_t*>   ux1 { "ux1", 2 };
  array_t<real_t*>   ux2 { "ux2", 2 };
  array_t<real_t*>   ux3 { "ux3", 2 };
  array_t<real_t*>   phi { "phi", 2 };
  array_t<real_t*>   weight { "weight", 2 };
  array_t<short*>    tag { "tag", 2 };
  const float        mass        = 1.0;
  const float        charge      = 1.0;
  const bool         use_weights = false;
  const real_t       inv_n0      = 1.0;

  put_value<int>(i1, 5, 0);
  put_value<int>(i2, 5, 0);
  put_value<int>(i3, 5, 0);
  put_value<prtldx_t>(dx1, (prtldx_t)(0.15), 0);
  put_value<prtldx_t>(dx2, (prtldx_t)(0.85), 0);
  put_value<prtldx_t>(dx3, (prtldx_t)(0.25), 0);
  put_value<real_t>(ux1, (real_t)(1.0), 0);
  put_value<real_t>(ux2, (real_t)(-2.0), 0);
  put_value<real_t>(ux3, (real_t)(3.0), 0);
  put_value<short>(tag, ParticleTag::alive, 0);

  put_value<int>(i1, 5, 1);
  put_value<int>(i2, 5, 1);
  put_value<int>(i3, 5, 1);
  put_value<prtldx_t>(dx1, (prtldx_t)(0.15), 1);
  put_value<prtldx_t>(dx2, (prtldx_t)(0.85), 1);
  put_value<prtldx_t>(dx3, (prtldx_t)(0.25), 1);
  put_value<real_t>(ux1, (real_t)(1.0), 1);
  put_value<real_t>(ux2, (real_t)(-2.0), 1);
  put_value<real_t>(ux3, (real_t)(3.0), 1);
  put_value<short>(tag, ParticleTag::alive, 1);

  // Particle boundaries
  auto boundaries = boundaries_t<PrtlBC> {};
  boundaries      = {
    {PrtlBC::PERIODIC, PrtlBC::PERIODIC},
    {PrtlBC::PERIODIC, PrtlBC::PERIODIC},
    {PrtlBC::PERIODIC, PrtlBC::PERIODIC}
  };

  auto pgen = Pgen {};
  // clang-format off
  auto pusher = Pusher_kernel<Minkowski<Dim::_3D>, Pgen, Boris_GCA_t, false>(
                              emfield, i1, i2, i3, i1, i2, i3, dx1, dx2, dx3,
                                       dx1, dx2, dx3, ux1, ux2, ux3, phi, tag, 
                                       metric, pgen,
                                       (real_t)0.0, coeff, dt, nx1, nx2, nx3, boundaries, 
                                       (real_t)10.0, (real_t)1.0, (real_t)1.0);
  // clang-format on
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;

    testGCAPusher<SimEngine::SRPIC, Minkowski<Dim::_3D>>(
      {
        10,
        10,
        10
    },
      { { 0.0, 10.0 }, { 0.0, 10.0 }, { 0.0, 10.0 } },
      {});

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}