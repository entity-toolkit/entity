#include "kernels/particle_moments.hpp"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/formatting.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"

#include "framework/containers/particles.h"
#include "kernels/reduced_stats.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <iostream>
#include <limits>
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
void put_value(array_t<T*>& arr, T v, prtlidx_t p) {
  auto h = Kokkos::create_mirror_view(arr);
  Kokkos::deep_copy(h, arr);
  h(p) = v;
  Kokkos::deep_copy(arr, h);
}

inline static constexpr auto epsilon = std::numeric_limits<real_t>::epsilon();

template <SimEngine::type S, typename M>
void testParticleMoments(const std::vector<ncells_t>&         res,
                         const boundaries_t<real_t>&          ext,
                         const std::map<std::string, real_t>& params = {},
                         const real_t                         acc    = ONE) {
  static_assert(M::Dim == 2);
  errorIf(res.size() != M::Dim, "res.size() != M::Dim");

  boundaries_t<real_t> extent;
  if constexpr (M::CoordType == Coord::Cartesian) {
    extent = ext;
  } else {
    extent = {
      ext[0],
      { ZERO, constant::PI }
    };
  }

  M metric { res, extent, params };

  const auto nx1 = res[0];
  const auto nx2 = res[1];

  ndfield_t<Dim::_2D, 3> buff { "buff", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };
  Particles<M::Dim, M::CoordType> particles { 1u,
                                              "test",
                                              1.0f,
                                              1.0f,
                                              10,
                                              0,
                                              0,
                                              ParticlePusher::NONE,
                                              false,
                                              RadiativeDrag::NONE,
                                              EmissionType::NONE,
                                              0,
                                              0 };
  const bool   use_weights = false;
  const real_t inv_n0      = 1.0;

  put_value<int>(particles.i1, 5, 0);
  put_value<int>(particles.i2, 4, 0);
  put_value<prtldx_t>(particles.dx1, (prtldx_t)(0.15), 0);
  put_value<prtldx_t>(particles.dx2, (prtldx_t)(0.85), 0);
  put_value<real_t>(particles.ux1, (real_t)(1.0), 0);
  put_value<real_t>(particles.ux2, (real_t)(-2.0), 0);
  put_value<real_t>(particles.ux3, (real_t)(3.0), 0);
  put_value<short>(particles.tag, ParticleTag::alive, 0);
  put_value<real_t>(particles.weight, 1.0, 0);

  put_value<int>(particles.i1, 2, 4);
  put_value<int>(particles.i2, 2, 4);
  put_value<prtldx_t>(particles.dx1, (prtldx_t)(0.22), 4);
  put_value<prtldx_t>(particles.dx2, (prtldx_t)(0.55), 4);
  put_value<real_t>(particles.ux1, (real_t)(-3.0), 4);
  put_value<real_t>(particles.ux2, (real_t)(2.0), 4);
  put_value<real_t>(particles.ux3, (real_t)(-1.0), 4);
  put_value<short>(particles.tag, ParticleTag::alive, 4);
  put_value<real_t>(particles.weight, 1.0, 4);

  auto boundaries = boundaries_t<FldsBC> {};
  if constexpr (M::CoordType != Coord::Cartesian) {
    boundaries = {
      { FldsBC::CUSTOM, FldsBC::CUSTOM },
      {   FldsBC::AXIS,   FldsBC::AXIS }
    };
  }

  const std::vector<uint8_t> comp1 { 0, 1 };
  const std::vector<uint8_t> comp2 { 0, 2 };
  const std::vector<uint8_t> comp3 { 0, 3 };
  const uint8_t              window = 1;

  auto scatter_buff = Kokkos::Experimental::create_scatter_view(buff);
  Kokkos::parallel_for(
    "ParticleMoments",
    10,
    kernel::ParticleMoments_kernel<S, M, FldsID::T, 3>(comp1,
                                                       scatter_buff,
                                                       0,
                                                       particles,
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
                                                       particles,
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
                                                       particles,
                                                       use_weights,
                                                       metric,
                                                       boundaries,
                                                       nx2,
                                                       inv_n0,
                                                       window));

  real_t n = ZERO, npart = ZERO, rho = ZERO, t00 = ZERO;
  Kokkos::parallel_reduce(
    "ReducedParticleMoments",
    10,
    kernel::ReducedParticleMoments_kernel<S, M, StatsID::N>({},
                                                            particles,
                                                            use_weights,
                                                            metric),
    n);

  Kokkos::parallel_reduce(
    "ReducedParticleMoments",
    10,
    kernel::ReducedParticleMoments_kernel<S, M, StatsID::Npart>({},
                                                                particles,
                                                                use_weights,
                                                                metric),
    npart);
  Kokkos::parallel_reduce(
    "ReducedParticleMoments",
    10,
    kernel::ReducedParticleMoments_kernel<S, M, StatsID::Rho>({},
                                                              particles,
                                                              use_weights,
                                                              metric),
    rho);
  Kokkos::parallel_reduce(
    "ReducedParticleMoments",
    10,
    kernel::ReducedParticleMoments_kernel<S, M, StatsID::T>({ 0u, 0u },
                                                            particles,
                                                            use_weights,
                                                            metric),
    t00);
  Kokkos::Experimental::contribute(buff, scatter_buff);

  auto i1_h = Kokkos::create_mirror_view(particles.i1);
  auto i2_h = Kokkos::create_mirror_view(particles.i2);
  Kokkos::deep_copy(i1_h, particles.i1);
  Kokkos::deep_copy(i2_h, particles.i2);

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
    const real_t gammaSQR_1 = ONE + v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
    const real_t gammaSQR_2 = ONE + v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2];

    const real_t gammaSQR_1_expect = 15.0;
    const real_t gammaSQR_2_expect = 15.0;

    const real_t n_expect   = 2.0;
    const real_t t00_expect = static_cast<real_t>(2.0 * math::sqrt(15.0));

    errorIf(not cmp::AlmostEqual_host(gammaSQR_1, gammaSQR_1_expect, epsilon * acc),
            fmt::format("wrong gamma_1 %.12e %.12e for %dD %s",
                        gammaSQR_1,
                        gammaSQR_1_expect,
                        metric.Dim,
                        metric.Label));
    errorIf(not cmp::AlmostEqual_host(gammaSQR_2, gammaSQR_2_expect, epsilon * acc),
            fmt::format("wrong gamma_2 %.12e %.12e for %dD %s",
                        gammaSQR_2,
                        gammaSQR_2_expect,
                        metric.Dim,
                        metric.Label));

    errorIf(not cmp::AlmostEqual_host(n, n_expect, epsilon * acc),
            fmt::format("wrong n reduction %.12e %.12e for %dD %s",
                        n,
                        n_expect,
                        metric.Dim,
                        metric.Label));
    errorIf(not cmp::AlmostEqual_host(npart, n_expect, epsilon * acc),
            fmt::format("wrong npart reduction %.12e %.12e for %dD %s",
                        npart,
                        n_expect,
                        metric.Dim,
                        metric.Label));
    errorIf(not cmp::AlmostEqual_host(rho, n_expect, epsilon * acc),
            fmt::format("wrong rho reduction %.12e %.12e for %dD %s",
                        rho,
                        n_expect,
                        metric.Dim,
                        metric.Label));
    errorIf(not cmp::AlmostEqual_host(t00, t00_expect, epsilon * acc),
            fmt::format("wrong t00 reduction %.12e %.12e for %dD %s",
                        t00,
                        t00_expect,
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
      {},
      10);

  } catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
