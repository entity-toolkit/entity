#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/spatial_dist.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct Firehose : public arch::EnergyDistribution<S, M> {
    Firehose(const M& metric, real_t time, real_t period, real_t vmax)
      : arch::EnergyDistribution<S, M> { metric }
      , phase { (real_t)(constant::TWO_PI)*time / period }
      , vmax { vmax } {}

    Inline void operator()(const coord_t<M::Dim>&,
                           vec_t<Dim::_3D>& v_Ph,
                           spidx_t) const override {
      v_Ph[0] = vmax * math::cos(phase);
      v_Ph[1] = vmax * math::sin(phase);
    }

  private:
    const real_t phase, vmax;
  };

  template <SimEngine::type S, class M>
  struct PointDistribution : public arch::SpatialDistribution<S, M> {
    PointDistribution(const M& metric, real_t x1c, real_t x2c, real_t dr)
      : arch::SpatialDistribution<S, M> { metric }
      , x1c { x1c }
      , x2c { x2c }
      , dr { dr } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t override {
      return math::exp(-(SQR(x_Ph[0] - x1c) + SQR(x_Ph[1] - x2c)) / SQR(dr));
    }

  private:
    const real_t x1c, x2c, dr;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions = traits::compatible_with<Dim::_2D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t period, vmax, x1c, x2c, dr, rate;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>&)
      : arch::ProblemGenerator<S, M> { p }
      , period { params.template get<real_t>("setup.period", 1.0) }
      , vmax { params.template get<real_t>("setup.vmax", 1.0) }
      , x1c { params.template get<real_t>("setup.x1c", 0.0) }
      , x2c { params.template get<real_t>("setup.x2c", 0.0) }
      , dr { params.template get<real_t>("setup.dr", 0.1) }
      , rate { params.template get<real_t>("setup.rate", 1.0) } {}

    void CustomPostStep(std::size_t, long double time, Domain<S, M>& domain) {
      const auto energy_dist  = Firehose<S, M>(domain.mesh.metric,
                                              (real_t)time,
                                              period,
                                              vmax);
      const auto spatial_dist = PointDistribution<S, M>(domain.mesh.metric,
                                                        x1c,
                                                        x2c,
                                                        dr);
      const auto injector = arch::NonUniformInjector<S, M, Firehose, PointDistribution>(
        energy_dist,
        spatial_dist,
        { 1, 2 });

      arch::InjectNonUniform<S, M, arch::NonUniformInjector<S, M, Firehose, PointDistribution>>(
        params,
        domain,
        injector,
        rate);
    }
  };

} // namespace user

#endif
