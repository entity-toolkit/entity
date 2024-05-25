#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/spatial_dist.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct Beam : public arch::EnergyDistribution<S, M> {
    Beam(const M& metric) : arch::EnergyDistribution<S, M> { metric } {}

    Inline void operator()(const coord_t<M::Dim>&,
                           vec_t<Dim::_3D>& v_Ph,
                           unsigned short   sp) const override {
      if (sp == 1) {
        v_Ph[0] = 0.0;
        v_Ph[1] = 10.0;
      } else {
        v_Ph[0] = -1.0;
        v_Ph[1] = 10.0;
      }
    }
  };

  template <SimEngine::type S, class M>
  struct PointDistribution : public arch::SpatialDistribution<S, M> {
    PointDistribution(const M& metric)
      : arch::SpatialDistribution<S, M> { metric } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t override {
      return (x_Ph[0] > 1.9 and x_Ph[0] < 2.1 and x_Ph[1] > 1.2 and x_Ph[1] < 1.3)
               ? ONE
               : ZERO;
    }
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Spherical, Metric::QSpherical>::value
    };
    static constexpr auto dimensions { traits::compatible_with<Dim::_2D>::value };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p) {}

    inline void InitPrtls(Domain<S, M>& domain) {
      const auto energy_dist  = Beam<S, M>(domain.mesh.metric);
      const auto spatial_dist = PointDistribution<S, M>(domain.mesh.metric);
      const auto injector = arch::NonUniformInjector<S, M, Beam, PointDistribution>(
        energy_dist,
        spatial_dist,
        { 1, 2 });

      arch::InjectNonUniform<S, M, arch::NonUniformInjector<S, M, Beam, PointDistribution>>(
        params,
        domain,
        injector,
        1.0);
    }
  };

} // namespace user

#endif
