#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"

#include "archetypes/particle_injector.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct DeltaDistribution {
    const real_t         photon_energy0;
    random_number_pool_t random_pool;

    DeltaDistribution(real_t photon_energy0, random_number_pool_t& random_pool)
      : photon_energy0 { photon_energy0 }
      , random_pool { random_pool } {}

    Inline void operator()(const coord_t<D>&, vec_t<Dim::_3D>& v) const {
      auto gen  = random_pool.get_state();
      auto rnd1 = Random<real_t>(gen);
      auto rnd2 = Random<real_t>(gen);
      random_pool.free_state(gen);
      // random direction
      const auto phi = static_cast<real_t>(constant::TWO_PI) * rnd1;
      const auto ct  = 2.0 * rnd2 - 1.0;
      const auto st  = math::sqrt(1.0 - ct * ct);
      v[0]           = photon_energy0 * st * math::cos(phi);
      v[1]           = photon_energy0 * st * math::sin(phi);
      v[2]           = photon_energy0 * ct;
    }
  };

  template <SimEngine::type S, class M>
  struct PGen {

    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D> {}
    };

    const SimulationParams& params;
    const Metadomain<S, M>& metadomain;

    PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : params { p }
      , metadomain { m } {}

    void InitPrtls(Domain<S, M>& domain) {
      const auto temperature = params.template get<real_t>("setup.temperature");
      arch::InjectUniformMaxwellian(params, domain, ONE, temperature, { 1u, 2u });

      auto delta = DeltaDistribution<M::Dim> { params.template get<real_t>(
                                                 "setup.photon_energy"),
                                               domain.random_pool() };
      arch::InjectUniform<S, M, decltype(delta)>(params, domain, 3u, delta, ONE);
    }
  };

} // namespace user

#endif
