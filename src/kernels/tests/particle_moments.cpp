#include "kernels/particle_moments.hpp"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/formatting.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

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

template <SimEngine::type S, typename M>
void testParticleMoments(const std::vector<std::size_t>&      res,
                         const boundaries_t<real_t>&          ext,
                         const std::map<std::string, real_t>& params = {},
                         const real_t                         acc    = ONE) {
  static_assert(M::Dim == 2);
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

  const auto nx1 = res[0];
  const auto nx2 = res[1];

  ndfield_t<Dim::_2D, 3> buff { "buff", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };
  array_t<int*>      i1 { "i1", 10 };
  array_t<int*>      i2 { "i2", 10 };
  array_t<int*>      i3 { "i3", 0 };
  array_t<prtldx_t*> dx1 { "dx1", 10 };
  array_t<prtldx_t*> dx2 { "dx2", 10 };
  array_t<prtldx_t*> dx3 { "dx3", 0 };
  array_t<real_t*>   ux1 { "ux1", 10 };
  array_t<real_t*>   ux2 { "ux2", 10 };
  array_t<real_t*>   ux3 { "ux3", 10 };
  array_t<real_t*>   phi { "phi", 10 };
  array_t<real_t*>   weight { "weight", 10 };
  array_t<short*>    tag { "tag", 10 };
  const float        mass        = 1.0;
  const float        charge      = 1.0;
  const bool         use_weights = false;
  const real_t       inv_n0      = 1.0;

  put_value<int>(i1, 5, 0);
  put_value<int>(i2, 4, 0);
  put_value<prtldx_t>(dx1, (prtldx_t)(0.15), 0);
  put_value<prtldx_t>(dx2, (prtldx_t)(0.85), 0);
  put_value<real_t>(ux1, (real_t)(1.0), 0);
  put_value<real_t>(ux2, (real_t)(-2.0), 0);
  put_value<real_t>(ux3, (real_t)(3.0), 0);
  put_value<short>(tag, ParticleTag::alive, 0);
  put_value<real_t>(weight, 1.0, 0);

  put_value<int>(i1, 2, 4);
  put_value<int>(i2, 2, 4);
  put_value<prtldx_t>(dx1, (prtldx_t)(0.22), 4);
  put_value<prtldx_t>(dx2, (prtldx_t)(0.55), 4);
  put_value<real_t>(ux1, (real_t)(-3.0), 4);
  put_value<real_t>(ux2, (real_t)(2.0), 4);
  put_value<real_t>(ux3, (real_t)(-1.0), 4);
  put_value<short>(tag, ParticleTag::alive, 4);
  put_value<real_t>(weight, 1.0, 4);

  auto boundaries = boundaries_t<FldsBC> {};
  if constexpr (M::CoordType != Coord::Cart) {
    boundaries = {
      {FldsBC::CUSTOM, FldsBC::CUSTOM},
      {  FldsBC::AXIS,   FldsBC::AXIS}
    };
  }

  const std::vector<unsigned short> comp1 { 0, 1 };
  const std::vector<unsigned short> comp2 { 0, 2 };
  const std::vector<unsigned short> comp3 { 0, 3 };
  const unsigned short              window = 1;

  auto scatter_buff = Kokkos::Experimental::create_scatter_view(buff);
  Kokkos::parallel_for(
    "ParticleMoments",
    10,
    kernel::ParticleMoments_kernel<S, M, FldsID::T, 3>(comp1,
                                                       scatter_buff,
                                                       0,
                                                       i1,
                                                       i2,
                                                       i3,
                                                       dx1,
                                                       dx2,
                                                       dx3,
                                                       ux1,
                                                       ux2,
                                                       ux3,
                                                       phi,
                                                       weight,
                                                       tag,
                                                       mass,
                                                       charge,
                                                       use_weights,
                                                       metric,
                                                       boundaries,
                                                       nx2,
                                                       inv_n0,
                                                       window));
  Kokkos::parallel_for(
    "ParticleMoments",
    10,
    kernel::ParticleMoments_kernel<S, M, FldsID::T, 3>(comp2,
                                                       scatter_buff,
                                                       1,
                                                       i1,
                                                       i2,
                                                       i3,
                                                       dx1,
                                                       dx2,
                                                       dx3,
                                                       ux1,
                                                       ux2,
                                                       ux3,
                                                       phi,
                                                       weight,
                                                       tag,
                                                       mass,
                                                       charge,
                                                       use_weights,
                                                       metric,
                                                       boundaries,
                                                       nx2,
                                                       inv_n0,
                                                       window));
  Kokkos::parallel_for(
    "ParticleMoments",
    10,
    kernel::ParticleMoments_kernel<S, M, FldsID::T, 3>(comp3,
                                                       scatter_buff,
                                                       2,
                                                       i1,
                                                       i2,
                                                       i3,
                                                       dx1,
                                                       dx2,
                                                       dx3,
                                                       ux1,
                                                       ux2,
                                                       ux3,
                                                       phi,
                                                       weight,
                                                       tag,
                                                       mass,
                                                       charge,
                                                       use_weights,
                                                       metric,
                                                       boundaries,
                                                       nx2,
                                                       inv_n0,
                                                       window));
  Kokkos::Experimental::contribute(buff, scatter_buff);

  auto i1_h = Kokkos::create_mirror_view(i1);
  auto i2_h = Kokkos::create_mirror_view(i2);
  Kokkos::deep_copy(i1_h, i1);
  Kokkos::deep_copy(i2_h, i2);

  const auto h1 = metric.sqrt_det_h({ static_cast<real_t>(i1_h(0)) + HALF,
                                      static_cast<real_t>(i2_h(0)) + HALF });
  const auto h2 = metric.sqrt_det_h({ static_cast<real_t>(i1_h(4)) + HALF,
                                      static_cast<real_t>(i2_h(4)) + HALF });

  {
    auto buff_h = Kokkos::create_mirror_view(buff);
    Kokkos::deep_copy(buff_h, buff);
    vec_t<Dim::_3D> v1 { ZERO };
    vec_t<Dim::_3D> v2 { ZERO };
    for (unsigned int idx1 = 0; idx1 < nx1 + 2 * N_GHOSTS; ++idx1) {
      for (unsigned int idx2 = 0; idx2 < nx2 + 2 * N_GHOSTS; ++idx2) {
        if (idx1 < 6) {
          v2[0] += buff_h(idx1, idx2, 0) * h2;
          v2[1] += buff_h(idx1, idx2, 1) * h2;
          v2[2] += buff_h(idx1, idx2, 2) * h2;
        } else {
          v1[0] += buff_h(idx1, idx2, 0) * h1;
          v1[1] += buff_h(idx1, idx2, 1) * h1;
          v1[2] += buff_h(idx1, idx2, 2) * h1;
        }
      }
    }
    const real_t gamma_1 = math::sqrt(
      ONE + v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
    const real_t gamma_2 = math::sqrt(
      ONE + v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);

    const real_t gamma_1_expect = math::sqrt(1.0 + 1.0 + 4.0 + 9.0);
    const real_t gamma_2_expect = math::sqrt(1.0 + 9.0 + 4.0 + 1.0);

    errorIf(not cmp::AlmostEqual(gamma_1, gamma_1_expect, epsilon * acc),
            fmt::format("wrong gamma_1 %.8e %.8e for %dD %s",
                        gamma_1,
                        gamma_1_expect,
                        metric.Dim,
                        metric.Label));
    errorIf(not cmp::AlmostEqual(gamma_2, gamma_2_expect, epsilon * acc),
            fmt::format("wrong gamma_2 %.8e %.8e for %dD %s",
                        gamma_2,
                        gamma_2_expect,
                        metric.Dim,
                        metric.Label));
  }
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;

    testParticleMoments<SimEngine::SRPIC, Minkowski<Dim::_2D>>(
      {
        10,
        10
    },
      { { 0.0, 10.0 }, { 0.0, 10.0 } },
      {});

    testParticleMoments<SimEngine::SRPIC, Spherical<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 2.0 } },
      {},
      10);

    testParticleMoments<SimEngine::SRPIC, QSpherical<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 10.0 } },
      { { "r0", 0.0 }, { "h", 0.25 } },
      10);

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}