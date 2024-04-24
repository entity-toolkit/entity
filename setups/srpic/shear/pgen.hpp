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
  struct ShearDist : public arch::EnergyDistribution<S, M> {
    ShearDist(const M& metric, real_t v_max, real_t sx2)
      : arch::EnergyDistribution<S, M> { metric }
      , v_max { v_max }
      , kx2 { static_cast<real_t>(constant::TWO_PI) / sx2 } {}

    Inline void operator()(const coord_t<M::Dim>& x_Ph,
                           vec_t<Dim::_3D>&       v,
                           unsigned short         sp) const override {
      if (sp == 1) {
        v[0] = v_max * math::sin(kx2 * x_Ph[1]);
      } else {
        v[0] = -v_max * math::sin(kx2 * x_Ph[1]);
      }
    }

  private:
    const real_t v_max, kx2;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions = traits::compatible_with<Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t sx2;
    const real_t vmax;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , sx2 { global_domain.mesh().extent(in::x2).second -
              global_domain.mesh().extent(in::x2).first }
      , vmax { p.get<real_t>("setup.vmax", 0.01) } {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = ShearDist<S, M>(local_domain.mesh.metric, vmax, sx2);
      const auto injector = arch::ParticleInjector<S, M, ShearDist>(energy_dist,
                                                                    { 1, 2 });
      arch::InjectUniformNumberDensity<S, M, arch::ParticleInjector<S, M, ShearDist>>(
        params,
        local_domain,
        injector,
        1.0);
    }
  };

} // namespace user

#endif
