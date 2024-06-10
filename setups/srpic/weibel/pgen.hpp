#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions =
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t temp_1, temp_2;
    const real_t drift_u_1, drift_u_2;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , temp_1 { p.template get<real_t>("setup.temp_1") }
      , temp_2 { p.template get<real_t>("setup.temp_2") }
      , drift_u_1 { p.template get<real_t>("setup.drift_u_1") }
      , drift_u_2 { p.template get<real_t>("setup.drift_u_2") } {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist_1 = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                        local_domain.random_pool,
                                                        temp_1,
                                                        -drift_u_1,
                                                        in::x3);
      const auto energy_dist_2 = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                        local_domain.random_pool,
                                                        temp_2,
                                                        drift_u_2,
                                                        in::x3);
      const auto injector_1 = arch::UniformInjector<S, M, arch::Maxwellian>(
        energy_dist_1,
        { 1, 2 });
      const auto injector_2 = arch::UniformInjector<S, M, arch::Maxwellian>(
        energy_dist_2,
        { 3, 4 });
      arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(
        params,
        local_domain,
        injector_1,
        HALF);
      arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(
        params,
        local_domain,
        injector_2,
        HALF);
    }
  };

} // namespace user

#endif
