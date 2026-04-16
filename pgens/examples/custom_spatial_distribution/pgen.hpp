#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/traits.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension Dim>
  struct CustomSpatialDistribution {
    static constexpr auto D = Dim;

    CustomSpatialDistribution() {}

    // the only requirement for the spatial distribution is to have this operator()
    //   that takes in a position and returns the number density in that region (in units of n0)
    Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
      const auto ndens = ONE -
                         (SQR(x_Ph[1] - math::pow(math::abs(x_Ph[0]), THIRD)) +
                          SQR(x_Ph[0]));
      if (ndens < ZERO) {
        return ZERO;
      }
      return ndens;
    }
  };

  template <SimEngine S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    static constexpr auto engines {
      arch::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      arch::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      arch::traits::pgen::compatible_with<Dim::_2D, Dim::_3D> {}
    };

    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& metadomain)
      : arch::ProblemGenerator<S, M> { p } {}

    inline void InitPrtls(Domain<S, M>& domain) {
      const auto sdist = CustomSpatialDistribution<M::Dim> {};
      const auto edist = arch::Maxwellian<S, M> { domain.mesh.metric,
                                                  domain.random_pool(),
                                                  static_cast<real_t>(0.1) };

      // distributions are then passed to the nonuniform particle injector
      // function (same energy distribution is used for both species)
      arch::InjectNonUniform<S, M, decltype(edist), decltype(edist), decltype(sdist)>(
        params,
        domain,
        { 1u, 2u },
        { edist, edist },
        sdist,
        ONE);
    }
  };

} // namespace user

#endif
