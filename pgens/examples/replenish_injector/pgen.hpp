#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/spatial_dist.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct EMFields {
    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return 0.8;
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ONE;
    }
  };

  template <Dimension D>
  struct NonUniformTargetDensity {
    Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
      // example of a non-uniform target density that peaks at the center of the domain
      real_t r2 { ZERO };
      for (auto d = 0u; d < D; ++d) {
        r2 += SQR(x_Ph[d] - HALF);
      }
      return math::exp(-r2 / SQR(static_cast<real_t>(0.2)));
      //                                                ^
      //                        characteristic width of the density profile
    }
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_2D, Dim::_3D> {}
    };

    const SimulationParams& params;

    EMFields<D> init_flds;

    const std::string target_density;

    PGen(const SimulationParams& p, const Metadomain<S, M>& /*metadomain*/)
      : params { p }
      , target_density { params.template get<std::string>(
          "setup.target_density") } {}

    void CustomPostStep(timestep_t step, simtime_t /*time*/, Domain<S, M>& domain) {
      if (step % 100u != 0u) {
        return;
      }
      // perform replenishment and injection every 100 timesteps

      // compute the density of species #1 and #2
      // and save in the field buffer (index 0)
      arch::ComputeMomentWithSpecies<S, M, FldsID::N, 3>(params,
                                                         domain,
                                                         { 1u, 2u },
                                                         domain.fields.buff);

      const auto energy_dist = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(),
        0.2); // <-- target temperature for injection
      if (target_density == "uniform") {
        // pass the computed density to the replenisher
        const auto replenish_sdist = arch::spatial_dist::ReplenishUniform<M, 3>(
          domain.mesh.metric,
          domain.fields.buff,
          0u,   // <-- index in buff where the density is stored
          ONE); // <-- target density for replenishment
        arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(replenish_sdist)>(
          params,
          domain,
          { 1, 2 },
          { energy_dist, energy_dist },
          replenish_sdist,
          ONE);
      } else {
        const auto target_density_profile = NonUniformTargetDensity<D> {};
        const auto replenish_sdist =
          arch::spatial_dist::Replenish<M, 3, decltype(target_density_profile)>(
            domain.mesh.metric,
            domain.fields.buff,
            0u, // <-- index in buff where the density is stored
            target_density_profile,
            ONE); // <-- target density for replenishment
        arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(replenish_sdist)>(
          params,
          domain,
          { 1, 2 },
          { energy_dist, energy_dist },
          replenish_sdist,
          ONE);
      }
    }
  };

} // namespace user

#endif
