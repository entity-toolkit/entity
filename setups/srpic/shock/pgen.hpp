#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct DriftDist : public arch::EnergyDistribution<S, M> {
    DriftDist(const M& metric, real_t ux)
      : arch::EnergyDistribution<S, M> { metric }
      , ux { ux } {}

    Inline void operator()(const coord_t<M::Dim>&,
                           vec_t<Dim::_3D>& v,
                           unsigned short) const override {
      v[0] = -ux;
      v[1] = 0.1 * ux;
    }

  private:
    const real_t ux;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics { traits::compatible_with<Metric::Minkowski>::value };
    static constexpr auto dimensions {
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
    };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t drift_ux, temperature;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , drift_ux { p.template get<real_t>("setup.drift_ux") }
      , temperature { p.template get<real_t>("setup.temperature") } {}

    inline PGen() {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                      local_domain.random_pool,
                                                      temperature,
                                                      -drift_ux,
                                                      in::x1);
      const auto injector    = arch::UniformInjector<S, M, arch::Maxwellian>(
        energy_dist,
        { 1, 2 });
      arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(
        params,
        local_domain,
        injector,
        1.0);
      // const auto energy_dist = DriftDist<S, M>(local_domain.mesh.metric, drift_ux);
      // const auto injector = arch::UniformInjector<S, M, DriftDist>(energy_dist,
      //                                                              { 1, 2 });
      // arch::InjectUniform<S, M, arch::UniformInjector<S, M, DriftDist>>(params,
      //                                                                   local_domain,
      //                                                                   injector,
      //                                                                   1.0);
    }
  };

} // namespace user

#endif
