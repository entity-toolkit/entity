#include "enums.h"
#include "global.h"

#include "metrics/qspherical.h"
#include "metrics/spherical.h"
#include "kernels/deposition/currents/global.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

using namespace ntt;

namespace {
  constexpr int nr = 32, nt = 1024;
  constexpr real_t guard = 1e-5;

  template <typename T>
  auto scalarArray(const char* name, T value) -> array_t<T*> {
    array_t<T*> result(name, 1);
    Kokkos::deep_copy(result, value);
    return result;
  }

  template <typename T>
  auto scalarValue(const array_t<T*>& arr) -> T {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), arr)(0);
  }

  void require(bool ok, const std::string& msg) {
    if (!ok) {
      throw std::runtime_error(msg);
    }
  }

  // Use the rounded endpoint representation seen in the SR position pusher:
  // i2 = nt - 1, dx2 = 1. The temporary transform must not rewrite it.
  auto particle(int j, prtldx_t dy, bool azimuthal) -> ParticleArrays {
    ParticleArrays p;
    p.i1 = scalarArray<int>("i1", 8);
    p.i2 = scalarArray<int>("i2", j);
    p.i1_prev = scalarArray<int>("i1_prev", 8);
    p.i2_prev = scalarArray<int>("i2_prev", j);
    p.dx1 = scalarArray<prtldx_t>("dx1", 0.25);
    p.dx2 = scalarArray<prtldx_t>("dx2", dy);
    p.dx1_prev = scalarArray<prtldx_t>("dx1_prev", 0.125);
    p.dx2_prev = scalarArray<prtldx_t>("dx2_prev", 0.5);
    const real_t phi = 0.37, uphi = azimuthal ? 0.2 : 0.0;
    p.ux1 = scalarArray<real_t>("ux1", -uphi * std::sin(phi));
    p.ux2 = scalarArray<real_t>("ux2", uphi * std::cos(phi));
    p.ux3 = scalarArray<real_t>("ux3", 0.1);
    p.phi = scalarArray<real_t>("phi", phi);
    p.weight = scalarArray<real_t>("weight", 0.75);
    p.tag = scalarArray<short>("tag", ParticleTag::alive);
    return p;
  }

  template <unsigned short O, typename M>
  auto deposit(const M& metric, const ParticleArrays& p) {
    ndfield_t<Dim::_2D, 3> J("J", nr + 2 * N_GHOSTS, nt + 2 * N_GHOSTS);
    Kokkos::deep_copy(J, ZERO);
    auto scatter = Kokkos::Experimental::create_scatter_view(J);
    Kokkos::parallel_for(
      "polar deposit", 1,
      kernel::DepositCurrents_kernel<SimEngine::SRPIC, M, O>(
        scatter, p, metric, -2.0, 0.25));
    Kokkos::Experimental::contribute(J, scatter);
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J);
  }

  template <unsigned short O, typename M>
  auto checkPosition(const M& metric, int j, prtldx_t dy,
                     const std::string& label) -> real_t {
    const auto p = particle(j, dy, true);
    const auto J = deposit<O>(metric, p);
    const auto Jzero = deposit<O>(metric, particle(j, dy, false));

    real_t sums[3] = { 0, 0, 0 };
    for (std::size_t i = 0; i < J.extent(0); ++i) {
      for (std::size_t k = 0; k < J.extent(1); ++k) {
        for (int c = 0; c < 3; ++c) {
          require(std::isfinite(J(i, k, c)), label + ": nonfinite current");
          sums[c] += J(i, k, c);
          if (c < 2) {
            require(J(i, k, c) == Jzero(i, k, c),
                    label + ": in-plane current depends on velocity guard");
          }
        }
      }
    }

    const real_t r = metric.template convert<1, Crd::Cd, Crd::Ph>(8.25);
    const real_t theta = metric.template convert<2, Crd::Cd, Crd::Ph>(
      static_cast<real_t>(j) + static_cast<real_t>(dy));
    const real_t safe_theta = std::clamp(theta, guard,
                                        static_cast<real_t>(constant::PI) - guard);
    // Independent physical expectation: shapes sum to unity and u_phi=0.2.
    const real_t expected = -1.5 * 0.2 /
                            (std::sqrt(1.05) * r * std::sin(safe_theta));
    const real_t tol = sizeof(real_t) == sizeof(double) ? 2e-8 : 0.03;
    require(std::abs(sums[2] / expected - 1) < tol,
            label + ": J_phi sum differs from bounded physical expectation; got " +
              std::to_string(sums[2]) + ", expected " + std::to_string(expected));
    require(std::abs(sums[0] - (-1.5 * 0.125 / 0.25)) < tol,
            label + ": radial trajectory current changed");
    require(std::abs(sums[1] - (-1.5 * (static_cast<real_t>(dy) - 0.5) / 0.25)) < tol,
            label + ": polar trajectory current changed");

    require(scalarValue(p.i1) == 8 && scalarValue(p.i2) == j &&
              scalarValue(p.dx1) == 0.25 && scalarValue(p.dx2) == dy &&
              scalarValue(p.i1_prev) == 8 && scalarValue(p.i2_prev) == j &&
              scalarValue(p.dx1_prev) == 0.125 && scalarValue(p.dx2_prev) == 0.5 &&
              scalarValue(p.phi) == static_cast<real_t>(0.37),
            label + ": stored particle trajectory changed");
    return sums[2];
  }

  template <unsigned short O, typename M>
  void checkMetric(const std::string& label,
                   const std::map<std::string, real_t>& params) {
    const real_t pi = constant::PI;
    // Match the angular cell size and local 1024-cell domains of the run.
    const M north({nr, nt}, {{1.0, 2.0}, {0.0, pi / 4}}, params);
    const M south({nr, nt}, {{1.0, 2.0}, {3 * pi / 4, pi}}, params);
    const auto prefix = label + ", O=" + std::to_string(O);
    const real_t n = checkPosition<O>(north, 0, 0, prefix + " north endpoint");
    const real_t s = checkPosition<O>(south, nt - 1, 1, prefix + " south endpoint");
    const real_t tol = sizeof(real_t) == sizeof(double) ? 2e-8 : 0.03;
    require(std::abs(n / s - 1) < tol, prefix + ": asymmetric polar limits");
    for (bool southern : {false, true}) {
      const auto& m = southern ? south : north;
      // Interior of guard, just outside it, and an ordinary off-axis point.
      for (real_t angle : {guard / 2, 2 * guard, static_cast<real_t>(0.01)}) {
        const real_t theta = southern ? pi - angle : angle;
        const real_t x = m.template convert<2, Crd::Ph, Crd::Cd>(theta);
        const int j = static_cast<int>(x);
        const prtldx_t dy = static_cast<prtldx_t>(x - j);
        checkPosition<O>(m, j, dy, prefix + " near-axis/off-axis");
      }
    }
    std::cout << "PASS " << prefix << " (8 positions)\n";
  }

  template <unsigned short O>
  void checkOrder() {
    checkMetric<O, metric::Spherical<Dim::_2D>>("spherical", {});
    checkMetric<O, metric::QSpherical<Dim::_2D>>(
      "qspherical h=0", {{"r0", 0.5}, {"h", 0.0}});
    checkMetric<O, metric::QSpherical<Dim::_2D>>(
      "qspherical h=0.25", {{"r0", 0.5}, {"h", 0.25}});
  }
} // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int result = 0;
  try {
    checkOrder<0>();
    checkOrder<1>();
    checkOrder<2>();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    result = 1;
  }
  Kokkos::finalize();
  return result;
}
