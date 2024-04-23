#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/comparators.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#include <vector>

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct SinEDist : public arch::EnergyDistribution<S, M> {
    SinEDist(const M& metric, real_t vx1_max, int nx1, real_t sx1)
      : arch::EnergyDistribution<S, M> { metric }
      , vx1_max { vx1_max }
      , kx1 { static_cast<real_t>(constant::TWO_PI) * static_cast<real_t>(nx1) /
              sx1 } {}

    Inline void operator()(const coord_t<M::Dim>& x_Ph,
                           vec_t<Dim::_3D>&       v,
                           unsigned short         sp) const override {
      if (sp == 1) {
        v[0] = vx1_max * math::sin(kx1 * x_Ph[0]);
      } else {
        v[0] = ZERO;
      }
    }

  private:
    const real_t vx1_max, kx1;
  };

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

    const real_t sx1;
    const real_t vmax;
    const int    nx1;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , sx1 { global_domain.mesh().extent(in::x1).second -
              global_domain.mesh().extent(in::x1).first }
      , vmax { p.get<real_t>("setup.vmax", 0.01) }
      , nx1 { p.get<int>("setup.nx1", 10) } {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = SinEDist<S, M>(local_domain.mesh.metric, vmax, nx1, sx1);
      const auto injector = arch::ParticleInjector<S, M, SinEDist>(energy_dist,
                                                                   { 1, 2 });
      arch::InjectUniformNumberDensity<S, M, arch::ParticleInjector<S, M, SinEDist>>(
        params,
        local_domain,
        injector,
        1.0);
    }
  };

} // namespace user

#endif
