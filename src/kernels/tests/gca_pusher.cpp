#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"

#include "kernels/particle_pusher_sr.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
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

template <SimEngine::type S, typename M>
void testGCAPusher(const std::vector<std::size_t>&      res,
                   const boundaries_t<real_t>&          ext,
                   const std::map<std::string, real_t>& params = {}) {
  static_assert(M::Dim == 3);
  errorIf(res.size() != M::Dim, "res.size() != M::Dim");

  boundaries_t<real_t> extent;
  extent = ext;

  M metric { res, extent, params };

  const int nx1 = res[0];
  const int nx2 = res[1];
  const int nx3 = res[2];

  auto coeff = real_t { 1.0 };
  auto dt    = real_t { 0.01 };

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
  array_t<int*>      i1_prev { "i1_prev", 2 };
  array_t<int*>      i2_prev { "i2_prev", 2 };
  array_t<int*>      i3_prev { "i3_prev", 2 };
  array_t<prtldx_t*> dx1 { "dx1", 2 };
  array_t<prtldx_t*> dx2 { "dx2", 2 };
  array_t<prtldx_t*> dx3 { "dx3", 2 };
  array_t<prtldx_t*> dx1_prev { "dx1_prev", 2 };
  array_t<prtldx_t*> dx2_prev { "dx2_prev", 2 };
  array_t<prtldx_t*> dx3_prev { "dx3_prev", 2 };
  array_t<real_t*>   ux1 { "ux1", 2 };
  array_t<real_t*>   ux2 { "ux2", 2 };
  array_t<real_t*>   ux3 { "ux3", 2 };
  array_t<real_t*>   phi { "phi", 2 };
  array_t<real_t*>   weight { "weight", 2 };
  array_t<short*>    tag { "tag", 2 };

  put_value<int>(i1, 5, 0);
  put_value<int>(i2, 5, 0);
  put_value<int>(i3, 5, 0);
  put_value<prtldx_t>(dx1, (prtldx_t)(0.15), 0);
  put_value<prtldx_t>(dx2, (prtldx_t)(0.85), 0);
  put_value<prtldx_t>(dx3, (prtldx_t)(0.25), 0);
  put_value<real_t>(ux1, (real_t)(1.0), 0);
  put_value<real_t>(ux2, (real_t)(-2.0), 0);
  put_value<real_t>(ux3, (real_t)(0.1), 0);
  put_value<short>(tag, ParticleTag::alive, 0);

  put_value<int>(i1, 5, 1);
  put_value<int>(i2, 5, 1);
  put_value<int>(i3, 5, 1);
  put_value<prtldx_t>(dx1, (prtldx_t)(0.15), 1);
  put_value<prtldx_t>(dx2, (prtldx_t)(0.85), 1);
  put_value<prtldx_t>(dx3, (prtldx_t)(0.25), 1);
  put_value<real_t>(ux1, (real_t)(1.0), 1);
  put_value<real_t>(ux2, (real_t)(-2.0), 1);
  put_value<real_t>(ux3, (real_t)(0.1), 1);
  put_value<short>(tag, ParticleTag::alive, 1);

  // Particle boundaries
  auto boundaries = boundaries_t<PrtlBC> {};
  boundaries      = {
    {PrtlBC::PERIODIC, PrtlBC::PERIODIC},
    {PrtlBC::PERIODIC, PrtlBC::PERIODIC},
    {PrtlBC::PERIODIC, PrtlBC::PERIODIC}
  };

  // clang-format off
  Kokkos::parallel_for(
    "pusher",
    1,
    kernel::sr::Pusher_kernel<Minkowski<Dim::_3D>>(PrtlPusher::BORIS,
                                                   true, false, kernel::sr::Cooling::None,
                                                   emfield,
                                                   1,
                                                   i1, i2, i3,
                                                   i1_prev, i2_prev, i3_prev,
                                                   dx1, dx2, dx3,
                                                   dx1_prev, dx2_prev, dx3_prev,
                                                   ux1, ux2, ux3,
                                                   phi, tag,
                                                   metric,
                                                   ZERO, coeff, dt,
                                                   nx1, nx2, nx3,
                                                   boundaries,
                                                   (real_t)100000.0, (real_t)1.0, ZERO));

  Kokkos::parallel_for(
    "pusher",
    CreateRangePolicy<Dim::_1D>({ 0 }, { 1 }),
    kernel::sr::Pusher_kernel<Minkowski<Dim::_3D>>(PrtlPusher::BORIS,
                                                   true, false, kernel::sr::Cooling::None,
                                                   emfield,
                                                   1,
                                                   i1, i2, i3,
                                                   i1_prev, i2_prev, i3_prev,
                                                   dx1, dx2, dx3,
                                                   dx1_prev, dx2_prev, dx3_prev,
                                                   ux1, ux2, ux3,
                                                   phi, tag,
                                                   metric,
                                                   ZERO, -coeff, dt,
                                                   nx1, nx2, nx3,
                                                   boundaries,
                                                   (real_t)100000.0, (real_t)1.0, ZERO));
  // clang-format on

  auto i1_prev_ = Kokkos::create_mirror_view(i1_prev);
  auto i2_prev_ = Kokkos::create_mirror_view(i2_prev);
  auto i3_prev_ = Kokkos::create_mirror_view(i3_prev);
  auto i1_      = Kokkos::create_mirror_view(i1);
  auto i2_      = Kokkos::create_mirror_view(i2);
  auto i3_      = Kokkos::create_mirror_view(i3);
  Kokkos::deep_copy(i1_prev_, i1_prev);
  Kokkos::deep_copy(i2_prev_, i2_prev);
  Kokkos::deep_copy(i3_prev_, i3_prev);
  Kokkos::deep_copy(i1_, i1);
  Kokkos::deep_copy(i2_, i2);
  Kokkos::deep_copy(i3_, i3);

  auto dx1_prev_ = Kokkos::create_mirror_view(dx1_prev);
  auto dx2_prev_ = Kokkos::create_mirror_view(dx2_prev);
  auto dx3_prev_ = Kokkos::create_mirror_view(dx3_prev);
  auto dx1_      = Kokkos::create_mirror_view(dx1);
  auto dx2_      = Kokkos::create_mirror_view(dx2);
  auto dx3_      = Kokkos::create_mirror_view(dx3);
  Kokkos::deep_copy(dx1_prev_, dx1_prev);
  Kokkos::deep_copy(dx2_prev_, dx2_prev);
  Kokkos::deep_copy(dx3_prev_, dx3_prev);
  Kokkos::deep_copy(dx1_, dx1);
  Kokkos::deep_copy(dx2_, dx2);
  Kokkos::deep_copy(dx3_, dx3);

  auto disx = i1_[0] + dx1_[0] - i1_prev_[0] - dx1_prev_[0];
  auto disy = i2_[0] + dx2_[0] - i2_prev_[0] - dx2_prev_[0];
  auto disz = i3_[0] + dx3_[0] - i3_prev_[0] - dx3_prev_[0];

  auto disdotB = (disx * 0.22 + disy * 0.44 + disz * 0.66) /
                 (0.823165 * math::sqrt(SQR(disx) + SQR(disy) + SQR(disz)));

  printf("%.12e \n", (1 - math::abs(disdotB)));

  disx = i1_[1] + dx1_[1] - i1_prev_[1] - dx1_prev_[1];
  disy = i2_[1] + dx2_[1] - i2_prev_[1] - dx2_prev_[1];
  disz = i3_[1] + dx3_[1] - i3_prev_[1] - dx3_prev_[1];

  disdotB = (disx * 0.22 + disy * 0.44 + disz * 0.66) /
            (0.823165 * math::sqrt(SQR(disx) + SQR(disy) + SQR(disz)));

  printf("%.12e \n", (1 - math::abs(disdotB)));
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
