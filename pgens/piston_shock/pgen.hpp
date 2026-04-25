#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"

#include "archetypes/piston.h"
#include "archetypes/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#include <vector>

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    // compatibility traits for the problem generator
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

    // domain properties
    const real_t global_xmin, global_xmax;

    // plasma
    const real_t temperature, temperature_ratio;
    // piston properties
    const real_t piston_velocity, piston_initial_position;

    PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : params { p }
      , metadomain { m }
      , global_xmin { metadomain.mesh().extent(in::x1).first }
      , global_xmax { metadomain.mesh().extent(in::x1).second }
      , piston_velocity { p.template get<real_t>("setup.piston_velocity", 0.0) }
      , piston_initial_position { p.template get<real_t>(
          "setup.piston_initial_position",
          0.0) }
      , temperature { p.template get<real_t>("setup.temperature", 0.0) }
      , temperature_ratio { p.template get<real_t>(
          "setup.temperature_ratio") } {}

    void InitPrtls(Domain<S, M>& local_domain) {
      real_t xg_min = global_xmin + piston_initial_position;
      real_t xg_max = global_xmax;

      // define box to inject into
      boundaries_t<real_t> box;
      // loop over all dimensions
      for (auto d { 0u }; d < (unsigned int)M::Dim; ++d) {
        // compute the range for the x-direction
        if (d == static_cast<decltype(d)>(in::x1)) {
          box.emplace_back(xg_min, xg_max);
        } else {
          // inject into full range in other directions
          box.push_back(Range::All);
        }
      }

      // define temperatures of species
      const auto temperatures = std::make_pair(temperature,
                                               temperature_ratio * temperature);
      // define drift speed of species
      const auto drifts = std::make_pair(std::vector<real_t> { ZERO, ZERO, ZERO },
                                         std::vector<real_t> { ZERO, ZERO, ZERO });

      // inject particles
      arch::InjectUniformMaxwellians<S, M>(params,
                                           local_domain,
                                           ONE,
                                           temperatures,
                                           { 1, 2 },
                                           drifts,
                                           false,
                                           box);
    }

    struct CustomPrtlUpdate {
      real_t piston_position_current; // current position of piston
      real_t piston_velocity_current;
      real_t xg_max;

      bool is_left;
      bool massive;

      Inline void operator()(index_t                          p,
                             const kernel::sr::PusherContext& ctx,
                             const kernel::sr::PusherBoundaries<M::Dim>&,
                             const kernel::PusherArrays& particles,
                             const M&                    metric) const {
        real_t piston_position_use;
        // make sure piston has not reached the right wall
        if (piston_position_current < xg_max) {
          piston_position_use = piston_position_current;
        } else {
          piston_position_use = xg_max;
        }

        if (arch::CrossesPiston<M>(p,
                                   ctx.dt,
                                   particles,
                                   metric,
                                   piston_position_use,
                                   piston_velocity_current,
                                   is_left)) {
          arch::PistonUpdate<M>(p,
                                ctx.dt,
                                particles,
                                metric,
                                piston_position_use,
                                piston_velocity_current,
                                massive);
        }
      }
    };

    template <class DOM>
    auto CustomParticleUpdate(simtime_t time, spidx_t /*sp*/, DOM& /*domain*/) const {
      return CustomPrtlUpdate { piston_initial_position +
                                  static_cast<real_t>(time) * piston_velocity,
                                piston_velocity,
                                metadomain.mesh().extent(in::x1).second,
                                true,
                                true };
    };
  };

} // namespace user

#endif // PROBLEM_GENERATOR_H