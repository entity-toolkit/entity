#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"

#include "kernels/particle_pusher_sr.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;

void check_value(unsigned int       t,
                 real_t             target,
                 real_t             value,
                 real_t             eps,
                 const std::string& msg) {
  const auto msg_ = fmt::format("%s: %e != %e @ %u", msg.c_str(), target, value, t);
  const auto diff = math::abs(target - value);
  const auto sum  = HALF * (math::abs(target) + math::abs(value));
  raise::ErrorIf(((sum > eps) and (diff / sum > eps)) or
                   ((sum <= eps) and (diff > eps / 10.0)),
                 msg_ + " " + fmt::format("%.12e, %.12e", diff, sum),
                 HERE);
}

template <typename T>
void put_value(array_t<T*>& arr, T v, index_t p) {
  auto h = Kokkos::create_mirror_view(arr);
  Kokkos::deep_copy(h, arr);
  h(p) = v;
  Kokkos::deep_copy(arr, h);
}

struct QED {
  array_t<int> cntr { "cntr" };

  QED() {}

  Inline void operator()(const vec_t<Dim::_3D>& u_xyz,
                         const vec_t<Dim::_3D>& e_xyz,
                         const vec_t<Dim::_3D>& b_xyz) const {
    const auto gamma = math::sqrt(ONE + NORM_SQR(u_xyz[0], u_xyz[1], u_xyz[2]));
    if (gamma > 1.5) {
      Kokkos::atomic_fetch_add(&cntr(), 1);
    }
  }

  auto get_cntr() const -> int {
    auto cntr_h = Kokkos::create_mirror_view(cntr);
    Kokkos::deep_copy(cntr_h, cntr);
    return cntr_h();
  }
};

template <SimEngine::type S, typename M>
void testQED(const std::vector<std::size_t>& res) {
  static_assert(M::Dim == 3);
  raise::ErrorIf(res.size() != M::Dim, "res.size() != M::Dim", HERE);

  M metric {
    res,
    { { 0.0, (real_t)(res[0]) }, { 0.0, (real_t)(res[1]) }, { 0.0, (real_t)(res[2]) } },
    {}
  };

  const int nx1 = res[0];
  const int nx2 = res[1];
  const int nx3 = res[2];

  const auto range_ext = CreateRangePolicy<Dim::_3D>(
    { 0, 0, 0 },
    { res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS });

  auto emfield = ndfield_t<Dim::_3D, 6> { "emfield",
                                          res[0] + 2 * N_GHOSTS,
                                          res[1] + 2 * N_GHOSTS,
                                          res[2] + 2 * N_GHOSTS };

  const real_t x1_0 = 1.15, x2_0 = 1.85, x3_0 = 1.25;
  const real_t ux1_0 = 3.02, ux2_0 = -6.2, ux3_0 = 3.1;
  // const real_t gamma_0 = math::sqrt(ONE + NORM_SQR(ux1_0, ux2_0, ux3_0));
  const real_t omegaB0 = 1.0;
  const real_t dt      = 0.01;

  Kokkos::parallel_for(
    "init 3D",
    range_ext,
    Lambda(index_t i1, index_t i2, index_t i3) {
      emfield(i1, i2, i3, em::ex1) = ZERO;
      emfield(i1, i2, i3, em::ex2) = ZERO;
      emfield(i1, i2, i3, em::ex3) = ZERO;
      emfield(i1, i2, i3, em::bx1) = ZERO;
      emfield(i1, i2, i3, em::bx2) = ZERO;
      emfield(i1, i2, i3, em::bx3) = ZERO;
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

  put_value<int>(i1, (int)(x1_0), 0);
  put_value<int>(i2, (int)(x2_0), 0);
  put_value<int>(i3, (int)(x3_0), 0);
  put_value<prtldx_t>(dx1, (prtldx_t)(x1_0 - (int)(x1_0)), 0);
  put_value<prtldx_t>(dx2, (prtldx_t)(x2_0 - (int)(x2_0)), 0);
  put_value<prtldx_t>(dx3, (prtldx_t)(x3_0 - (int)(x3_0)), 0);
  put_value<real_t>(ux1, ux1_0, 0);
  put_value<real_t>(ux2, ux2_0, 0);
  put_value<real_t>(ux3, ux3_0, 0);
  put_value<short>(tag, ParticleTag::alive, 0);

  put_value<int>(i1, (int)(x1_0), 1);
  put_value<int>(i2, (int)(x2_0), 1);
  put_value<int>(i3, (int)(x3_0), 1);
  put_value<prtldx_t>(dx1, (prtldx_t)(x1_0 - (int)(x1_0)), 1);
  put_value<prtldx_t>(dx2, (prtldx_t)(x2_0 - (int)(x2_0)), 1);
  put_value<prtldx_t>(dx3, (prtldx_t)(x3_0 - (int)(x3_0)), 1);
  put_value<real_t>(ux1, -ux1_0 / 100.0, 1);
  put_value<real_t>(ux2, -ux2_0 / 100.0, 1);
  put_value<real_t>(ux3, -ux3_0 / 100.0, 1);
  put_value<short>(tag, ParticleTag::alive, 1);

  // Particle boundaries
  auto boundaries = boundaries_t<PrtlBC> {};
  boundaries      = {
    { PrtlBC::PERIODIC, PrtlBC::PERIODIC },
    { PrtlBC::PERIODIC, PrtlBC::PERIODIC },
    { PrtlBC::PERIODIC, PrtlBC::PERIODIC }
  };

  const spidx_t sp { 1u };

  const real_t coeff = HALF * dt * omegaB0;

  const real_t eps = std::is_same_v<real_t, float> ? 1e-4 : 1e-6;

  const auto qed_process = QED {};

  for (auto t { 0u }; t < 100; ++t) {
    const real_t time = t * dt;

    // clang-format off
    auto pusher_params = kernel::sr::PusherParams<Minkowski<Dim::_3D>, kernel::sr::NoForce_t, decltype(qed_process)>(
                                                        PrtlPusher::BORIS,
                                                        kernel::sr::DisableGCA, kernel::sr::DisableExtForce, 
                                                        kernel::sr::Cooling::None,
                                                        emfield,
                                                        sp,
                                                        i1, i2, i3,
                                                        i1_prev, i2_prev, i3_prev,
                                                        dx1, dx2, dx3,
                                                        dx1_prev, dx2_prev, dx3_prev,
                                                        ux1, ux2, ux3,
                                                        phi, tag,
                                                        metric,
                                                        (simtime_t)time, coeff, dt,
                                                        nx1, nx2, nx3,
                                                        boundaries,
                                                        ZERO, ZERO, ZERO);
    pusher_params.qed = &qed_process;
    // clang-format on
    Kokkos::parallel_for(
      "pusher",
      2,
      kernel::sr::Pusher_kernel<Minkowski<Dim::_3D>, kernel::sr::NoForce_t, decltype(qed_process)>(
        pusher_params));
  }
  raise::ErrorIf(qed_process.get_cntr() != 100, "Wrong # of particles created", HERE);
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;

    testQED<SimEngine::SRPIC, Minkowski<Dim::_3D>>({ 10, 10, 10 });

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
