#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension Dim>
  struct CustomDistribution_1 {
    static constexpr auto D = Dim;

    CustomDistribution_1(random_number_pool_t& random_pool,
                         real_t                temperature,
                         real_t                drift_amplitude,
                         real_t                box_size)
      : random_pool { random_pool }
      , temperature { temperature }
      , drift_amplitude { drift_amplitude }
      , kx { static_cast<real_t>(constant::TWO_PI) / box_size } {}

    // the only requirement for the energy distribution is to have this operator()
    //   that takes in the particle position and velocity (by reference) and
    //   modifies (sets) the velocity according to the desired distribution
    Inline void operator()(const coord_t<D>& x_Ph, vec_t<Dim::_3D>& v) const {
      // sample a static 3D maxwellian + drift in x1 direction with sinusoidal spatial dependence
      // @NOTE: for relativistic drift, use the built-in drifting Maxwellian
      arch::JuttnerSinge(v, temperature, random_pool);
      v[0] += drift_amplitude * math::sin(x_Ph[0] * kx);
    }

    random_number_pool_t random_pool;
    const real_t         temperature, drift_amplitude, kx;
  };

  template <Dimension Dim>
  struct CustomDistribution_2 {
    static constexpr auto D = Dim;

    CustomDistribution_2(real_t drift_amplitude)
      : drift_amplitude { drift_amplitude } {}

    Inline void operator()(const coord_t<D>& x_Ph, vec_t<Dim::_3D>& v) const {
      // zero temperature + counterstreaming drifts in x2
      v[1] = drift_amplitude * ((x_Ph[0] < ZERO) ? ONE : -ONE);
    }

    const real_t drift_amplitude;
  };

  template <SimEngine S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D> {}
    };

    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const Metadomain<S, M>& metadomain;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& metadomain)
      : arch::ProblemGenerator<S, M> { p }
      , metadomain { metadomain } {}

    inline void InitPrtls(Domain<S, M>& domain) {
      const auto temperature = params.template get<real_t>("setup.temperature");
      const auto drift_amplitude = params.template get<real_t>(
        "setup.drift_amplitude");
      const auto box_size = metadomain.mesh().extent(in::x1).second -
                            metadomain.mesh().extent(in::x1).first;
      const auto edist1 = CustomDistribution_1<M::Dim> { domain.random_pool(),
                                                         temperature,
                                                         drift_amplitude,
                                                         box_size };
      const auto edist2 = CustomDistribution_2<M::Dim> { drift_amplitude };

      // distributions are then passed to the particle injector function
      arch::InjectUniform<S, M, decltype(edist1), decltype(edist2)>(
        params,
        domain,
        { 1u, 2u },
        { edist1, edist2 },
        ONE);
    }
  };

} // namespace user

#endif
