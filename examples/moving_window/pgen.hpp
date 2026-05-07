#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"

#include "archetypes/energy_dist.h"
#include "archetypes/moving_window.h"
#include "archetypes/particle_injector.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

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
      return math::exp(-r2 / SQR(static_cast<real_t>(0.1)));
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

    struct MovingWindow {
      int    pos_i { -1 };
      int    init_pos_i { -1 };
      real_t pos_di { ZERO };
      real_t vel_Cd { ZERO };

      void init(const M& metric, real_t global_x, real_t velocity) {
        const auto pos_Cd = metric.template convert<1, Crd::Ph, Crd::Cd>(global_x);
        pos_i      = static_cast<int>(pos_Cd + 1) - 1;
        init_pos_i = pos_i;
        pos_di     = pos_Cd - static_cast<real_t>(pos_i);
        vel_Cd = metric.template transform<1, Idx::XYZ, Idx::U>({}, velocity);
      }

      void update(real_t            dt,
                  int               shift,
                  Metadomain<S, M>& metadomain,
                  Domain<S, M>&     domain) {
        pos_di += vel_Cd * dt;
        pos_i  += static_cast<int>(pos_di >= ONE);
        pos_di -= static_cast<real_t>(pos_di >= ONE);
        if ((pos_i - init_pos_i) >= shift) {
          // move the window and all fields and particles in it
          arch::MoveWindow<S, M, in::x1>(domain, metadomain, shift);
          // update window index for next update
          pos_i -= shift;
        }
      }
    };

    MovingWindow moving_window;
    const real_t dt;

    PGen(const SimulationParams& p, Metadomain<S, M>& m)
      : params { p }
      , metadomain { m }
      , dt { params.template get<real_t>("algorithms.timestep.dt") } {}

    void InitPrtls(Domain<S, M>& domain) {
      const auto window_velocity = params.template get<real_t>(
        "setup.window_velocity",
        ZERO);
      moving_window.init(domain.mesh.metric, ZERO, window_velocity);
      // inject particles
      const auto energy_dist = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(),
        0.001,
        { window_velocity, ZERO, ZERO });

      const auto density_profile = NonUniformDensity<M::Dim> {};
      arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(density_profile)>(
        params,
        domain,
        { 1, 2 },
        { energy_dist, energy_dist },
        density_profile,
        ONE);
    }

    void CustomPostStep(timestep_t /*step*/, simtime_t /*time*/, Domain<S, M>& domain) {
      moving_window.update(dt, N_GHOSTS, metadomain, domain);
    }
  };

} // namespace user

#endif
