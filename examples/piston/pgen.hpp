#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/piston.h"
#include "framework/containers/particles.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "kernels/pushers/context.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct NonUniformDensity {
    Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
      // example of a non-uniform target density that peaks at the center of the domain
      real_t r2 { ZERO };
      for (auto d = 0u; d < D; ++d) {
        r2 += SQR(x_Ph[d] - 0.5);
      }
      return math::exp(-r2 / SQR(static_cast<real_t>(0.2)));
      //                                                ^
      //                        characteristic width of the density profile
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
      ::traits::pgen::compatible_with<Dim::_2D, Dim::_3D> {}
    };

    const SimulationParams& params;
    Metadomain<S, M>&       metadomain;
    const real_t            piston_velocity;

    PGen(const SimulationParams& p, Metadomain<S, M>& m)
      : params { p }
      , metadomain { m }
      , piston_velocity { params.template get<real_t>("setup.piston_velocity",
                                                      0.0) } {}

    void InitPrtls(Domain<S, M>& domain) {
      // inject particles
      const auto energy_dist = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(),
        0.2); // <-- target temperature for injection

      const auto density_profile = NonUniformDensity<M::Dim> {};
      arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(density_profile)>(
        params,
        domain,
        { 1, 2 },
        { energy_dist, energy_dist },
        density_profile,
        ONE);
    }

    struct CustomPrtlUpdate {
      real_t x_piston; // current position of piston
      real_t v_piston;
      real_t global_xmax;
      bool   is_left;
      bool   massive;

      Inline void operator()(prtlidx_t                        p,
                             const kernel::sr::PusherContext& ctx,
                             const kernel::sr::PusherBoundaries<M::Dim>&,
                             const ParticleArrays& particles,
                             const M&              metric) const {

        real_t piston_pos;
        if (x_piston > global_xmax) {
          // piston has moved beyond the domain, set position to a large value so that no particles interact with it
          piston_pos = global_xmax;
        } else {
          piston_pos = x_piston;
        }

        arch::Piston<M>(p, ctx.dt, particles, metric, piston_pos, v_piston, massive);
      }
    };

    template <class DOM>
    auto CustomParticleUpdate(simtime_t time, spidx_t /*sp*/, DOM& /*domain*/) const {
      return CustomPrtlUpdate { piston_velocity * static_cast<real_t>(time),
                                piston_velocity,
                                metadomain.mesh().extent(in::x1).second,
                                true,
                                true };
    };
  };
} // namespace user

#endif
