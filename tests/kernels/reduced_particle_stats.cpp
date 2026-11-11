#include "metrics/minkowski.h"

#include "kernels/reduced_stats.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace ntt;

void check(real_t actual, double expected, const std::string& label) {
  const double tolerance = 64 * std::numeric_limits<real_t>::epsilon() *
                           std::max(1.0, std::abs(expected));
  if (!std::isfinite(actual) || std::abs(double(actual) - expected) > tolerance) {
    throw std::runtime_error(label + ": actual=" + std::to_string(actual) +
                             ", expected=" + std::to_string(expected));
  }
}

template <Dimension D>
void testParticleStats(float mass, bool unit_weights, bool use_weights) {
  using M = metric::Minkowski<D>;
  // Non-unit cell volume catches accidental changes to the reduction measure.
  const M                        metric(std::vector<ncells_t>(D, 8),
                 boundaries_t<real_t>(D, { ZERO, FOUR }),
                                        {});
  Particles<D, Coord::Cartesian> particles(1,
                                           "stats",
                                           mass,
                                           mass == 0 ? 0 : -1,
                                           4,
                                           0,
                                           0,
                                           ParticlePusher::NONE,
                                           false,
                                           RadiativeDrag::NONE,
                                           EmissionType::NONE,
                                           0,
                                           0);
  const ParticleArrays           arrays = particles;
  Kokkos::parallel_for(
    "InitializeStatsParticles",
    4,
    Lambda(int p) {
      arrays.i1(p)  = 2;
      arrays.dx1(p) = HALF;
      if constexpr (D >= Dim::_2D) {
        arrays.i2(p)  = 2;
        arrays.dx2(p) = HALF;
      }
      if constexpr (D == Dim::_3D) {
        arrays.i3(p)  = 2;
        arrays.dx3(p) = HALF;
      }
      arrays.ux1(p)    = p == 0 ? THREE : p == 1 ? -ONE : ZERO;
      arrays.ux2(p)    = p == 1 ? TWO : p == 2 ? THREE : ZERO;
      arrays.ux3(p)    = p == 0   ? ZERO
                         : p == 1 ? -TWO
                         : p == 2 ? FOUR
                                  : real_t(100);
      arrays.weight(p) = unit_weights ? ONE
                         : p == 0     ? real_t(0.25)
                         : p == 1     ? TWO
                         : p == 2     ? ZERO
                                      : real_t(1000);
      arrays.tag(p)    = p == 3 ? ParticleTag::dead : ParticleTag::alive;
    });

  const double u[3][3] = {
    {  3, 0,  0 },
    { -1, 2, -2 },
    {  0, 3,  4 }
  };
  const double      weights[3] = { 0.25, 2, 0 };
  const double      dV         = std::pow(0.5, int(D));
  const std::string label      = "D=" + std::to_string(int(D)) +
                            " mass=" + std::to_string(mass) +
                            " unit=" + std::to_string(unit_weights) +
                            " weighted=" + std::to_string(use_weights);
  // All tensor components for unit-mass charges and photons. For other masses,
  // only T00 is in scope here; spatial-component mass conventions are unchanged.
  const int components = mass > 1 ? 1 : 4;
  for (uint8_t a = 0; a < components; ++a) {
    for (uint8_t b = 0; b < components; ++b) {
      double expected = 0;
      for (int p = 0; p < 3; ++p) {
        const double norm2 = u[p][0] * u[p][0] + u[p][1] * u[p][1] +
                             u[p][2] * u[p][2];
        const double energy = mass == 0 ? std::sqrt(norm2)
                                        : mass * std::sqrt(1 + norm2);
        const double weight = use_weights && !unit_weights ? weights[p] : 1;
        if (a == 0 && b == 0) {
          expected += weight * energy;
        } else if (a == 0) {
          expected += weight * u[p][b - 1];
        } else if (b == 0) {
          expected += weight * u[p][a - 1];
        } else {
          expected += weight * u[p][a - 1] * u[p][b - 1] / energy;
        }
      }
      real_t actual = ZERO;
      Kokkos::parallel_reduce(
        "ReducedParticleStressEnergy",
        4,
        kernel::ReducedParticleMoments_kernel<SimEngine::SRPIC, M, StatsID::T>(
          { a, b },
          particles,
          use_weights,
          metric),
        actual);
      check(actual,
            dV * expected,
            label + " T" + std::to_string(a) + std::to_string(b));
    }
  }
  real_t count = ZERO;
  Kokkos::parallel_reduce(
    "ReducedMacroParticleCount",
    4,
    kernel::ReducedParticleMoments_kernel<SimEngine::SRPIC, M, StatsID::Npart>(
      {},
      particles,
      use_weights,
      metric),
    count);
  check(count, 3, label + " Npart");

  Kokkos::deep_copy(particles.tag, short(ParticleTag::dead));
  real_t empty = ZERO;
  Kokkos::parallel_reduce(
    "ReducedDeadParticleEnergy",
    4,
    kernel::ReducedParticleMoments_kernel<SimEngine::SRPIC, M, StatsID::T>(
      { 0, 0 },
      particles,
      use_weights,
      metric),
    empty);
  check(empty, 0, label + " all dead");
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int result = 0;
  try {
    for (float mass : { 0.0f, 1.0f, 3.0f }) {
      for (bool unit_weights : { false, true }) {
        for (bool use_weights : { false, true }) {
          testParticleStats<Dim::_1D>(mass, unit_weights, use_weights);
          testParticleStats<Dim::_2D>(mass, unit_weights, use_weights);
          testParticleStats<Dim::_3D>(mass, unit_weights, use_weights);
        }
      }
    }
    std::cout
      << "PASS: 36 configurations, 396 tensor checks, 72 count/dead checks\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    result = 1;
  }
  Kokkos::finalize();
  return result;
}
