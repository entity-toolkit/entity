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

#include <vector>

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct Beam : public arch::EnergyDistribution<S, M> {
    Beam(const M&                   metric,
         const std::vector<real_t>& v1_vec,
         const std::vector<real_t>& v2_vec)
      : arch::EnergyDistribution<S, M> { metric } {
      std::copy(v1_vec.begin(), v1_vec.end(), v1);
      std::copy(v2_vec.begin(), v2_vec.end(), v2);
    }

    Inline void operator()(const coord_t<M::Dim>&,
                           vec_t<Dim::_3D>& v_Ph,
                           unsigned short   sp) const override {
      if (sp == 1) {
        v_Ph[0] = v1[0];
        v_Ph[1] = v1[1];
        v_Ph[2] = v1[2];
      } else {
        v_Ph[0] = v2[0];
        v_Ph[1] = v2[1];
        v_Ph[2] = v2[2];
      }
    }

  private:
    vec_t<Dim::_3D> v1;
    vec_t<Dim::_3D> v2;
  };

  template <SimEngine::type S, class M>
  struct PointDistribution : public arch::SpatialDistribution<S, M> {
    PointDistribution(const M&                   metric,
                      const std::vector<real_t>& xi_min,
                      const std::vector<real_t>& xi_max)
      : arch::SpatialDistribution<S, M> { metric } {
      std::copy(xi_min.begin(), xi_min.end(), x_min);
      std::copy(xi_max.begin(), xi_max.end(), x_max);
    }

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t override {
      auto fill = true;
      for (auto d = 0u; d < M::Dim; ++d) {
        fill &= x_Ph[d] > x_min[d] and x_Ph[d] < x_max[d];
      }
      return fill ? ONE : ZERO;
    }

  private:
    tuple_t<real_t, M::Dim> x_min;
    tuple_t<real_t, M::Dim> x_max;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Minkowski, Metric::Spherical, Metric::QSpherical>::value
    };
    static constexpr auto dimensions {
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
    };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const std::vector<real_t> xi_min;
    const std::vector<real_t> xi_max;
    const std::vector<real_t> v1;
    const std::vector<real_t> v2;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , xi_min { p.template get<std::vector<real_t>>("setup.xi_min") }
      , xi_max { p.template get<std::vector<real_t>>("setup.xi_max") }
      , v1 { p.template get<std::vector<real_t>>("setup.v1") }
      , v2 { p.template get<std::vector<real_t>>("setup.v2") } {}

    inline void InitPrtls(Domain<S, M>& domain) {
      const auto energy_dist  = Beam<S, M>(domain.mesh.metric, v1, v2);
      const auto spatial_dist = PointDistribution<S, M>(domain.mesh.metric,
                                                        xi_min,
                                                        xi_max);
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
