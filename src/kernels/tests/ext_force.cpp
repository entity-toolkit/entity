#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"
#include "utils/plog.h"

#include "metrics/minkowski.h"

#include "kernels/particle_pusher_sr.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Appenders/RollingFileAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>
#include <plog/Log.h>

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

struct Force {
  const std::vector<spidx_t> species { 1 };

  Force(real_t force) : force { force } {}

  Inline auto fx1(const spidx_t&,
                  const simtime_t&,
                  const coord_t<Dim::_3D>&) const -> real_t {
    return force * math::sin(ONE) * math::sin(ONE);
  }

  Inline auto fx2(const spidx_t&,
                  const simtime_t&,
                  const coord_t<Dim::_3D>&) const -> real_t {
    return force * math::sin(ONE) * math::cos(ONE);
  }

  Inline auto fx3(const spidx_t&,
                  const simtime_t&,
                  const coord_t<Dim::_3D>&) const -> real_t {
    return force * math::cos(ONE);
  }

private:
  const real_t force;
};

template <SimEngine::type S, typename M>
void testPusher(const std::vector<std::size_t>& res) {
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
  const real_t ux1_0 = 0.02, ux2_0 = -0.2, ux3_0 = 0.1;
  // const real_t gamma_0 = math::sqrt(ONE + NORM_SQR(ux1_0, ux2_0, ux3_0));
  const real_t omegaB0 = 1.0;
  const real_t dt      = 0.01;
  const real_t f_mag   = 0.01;

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
  put_value<real_t>(ux1, -ux1_0, 1);
  put_value<real_t>(ux2, -ux2_0, 1);
  put_value<real_t>(ux3, -ux3_0, 1);
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

  const auto ext_force = Force { f_mag };
  const auto force =
    kernel::sr::Force<Dim::_3D, Coord::Cart, decltype(ext_force), false> { ext_force };

  static plog::RollingFileAppender<plog::NttInfoFormatter> file_appender(
    "pusher_log.csv");
  plog::init(plog::verbose, &file_appender);
  PLOGD << "t,i1,i2,i3,dx1,dx2,dx3,ux1,ux2,ux3";

  for (auto t { 0u }; t < 100; ++t) {
    const real_t time = t * dt;

    // clang-format off
    Kokkos::parallel_for(
      "pusher",
      CreateRangePolicy<Dim::_1D>({0}, {2}),
      kernel::sr::Pusher_kernel<Minkowski<Dim::_3D>, decltype(force)>(PrtlPusher::BORIS,
                                                     false, true, kernel::sr::Cooling::None,
                                                     emfield,
                                                     sp,
                                                     i1, i2, i3,
                                                     i1_prev, i2_prev, i3_prev,
                                                     dx1, dx2, dx3,
                                                     dx1_prev, dx2_prev, dx3_prev,
                                                     ux1, ux2, ux3,
                                                     phi, tag,
                                                     metric, force,
                                                     (simtime_t)time, coeff, dt,
                                                     nx1, nx2, nx3,
                                                     boundaries,
                                                     ZERO, ZERO, ZERO));

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
    auto ux1_      = Kokkos::create_mirror_view(ux1);
    auto ux2_      = Kokkos::create_mirror_view(ux2);
    auto ux3_      = Kokkos::create_mirror_view(ux3);
    Kokkos::deep_copy(dx1_prev_, dx1_prev);
    Kokkos::deep_copy(dx2_prev_, dx2_prev);
    Kokkos::deep_copy(dx3_prev_, dx3_prev);
    Kokkos::deep_copy(dx1_, dx1);
    Kokkos::deep_copy(dx2_, dx2);
    Kokkos::deep_copy(dx3_, dx3);
    Kokkos::deep_copy(ux1_, ux1);
    Kokkos::deep_copy(ux2_, ux2);
    Kokkos::deep_copy(ux3_, ux3);

    PLOGD.printf("%e,%d,%d,%d,%e,%e,%e,%e,%e,%e",
                 time,
                 i1_(1),
                 i2_(1),
                 i3_(1),
                 dx1_( 1),
                 dx2_( 1),
                 dx3_( 1),
                 ux1_( 1),
                 ux2_( 1),
                 ux3_( 1));

    {
      const real_t ux1_expect = ux1_0 + (time + dt) * f_mag * std::sin(ONE) * std::sin(ONE);
      const real_t ux2_expect = ux2_0 + (time + dt) * f_mag * std::sin(ONE) * std::cos(ONE);
      const real_t ux3_expect = ux3_0 + (time + dt) * f_mag * std::cos(ONE);

      check_value(t, ux1_(0), ux1_expect, eps, "Particle #1 ux1");
      check_value(t, ux2_(0), ux2_expect, eps, "Particle #1 ux2");
      check_value(t, ux3_(0), ux3_expect, eps, "Particle #1 ux3");
    }

    {
      const real_t ux1_expect = -ux1_0 + (time + dt) * f_mag * std::sin(ONE) * std::sin(ONE);
      const real_t ux2_expect = -ux2_0 + (time + dt) * f_mag * std::sin(ONE) * std::cos(ONE);
      const real_t ux3_expect = -ux3_0 + (time + dt) * f_mag * std::cos(ONE);

      check_value(t, ux1_(1), ux1_expect, eps, "Particle #2 ux1");
      check_value(t, ux2_(1), ux2_expect, eps, "Particle #2 ux2");
      check_value(t, ux3_(1), ux3_expect, eps, "Particle #2 ux3");
    }

  }
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;

    testPusher<SimEngine::SRPIC, Minkowski<Dim::_3D>>({ 10, 10, 10 });

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
