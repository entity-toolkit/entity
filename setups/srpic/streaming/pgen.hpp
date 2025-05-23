#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

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

    using prmvec_t = std::vector<real_t>;

    prmvec_t drifts_in_x, drifts_in_y, drifts_in_z;
    prmvec_t densities, temperatures;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , drifts_in_x { p.template get<prmvec_t>("setup.drifts_in_x", prmvec_t {}) }
      , drifts_in_y { p.template get<prmvec_t>("setup.drifts_in_y", prmvec_t {}) }
      , drifts_in_z { p.template get<prmvec_t>("setup.drifts_in_z", prmvec_t {}) }
      , densities { p.template get<prmvec_t>("setup.densities", prmvec_t {}) }
      , temperatures { p.template get<prmvec_t>("setup.temperatures", prmvec_t {}) } {
      const auto nspec = p.template get<std::size_t>("particles.nspec");
      raise::ErrorIf(nspec % 2 != 0,
                     "Number of species must be even for this setup",
                     HERE);
      for (auto n = 0u; n < nspec; n += 2) {
        raise::ErrorIf(
          global_domain.species_params()[n].charge() !=
            -global_domain.species_params()[n + 1].charge(),
          "Charges of i-th and i+1-th species must be opposite for this setup",
          HERE);
      }
      for (auto* specs :
           { &drifts_in_x, &drifts_in_y, &drifts_in_z, &temperatures }) {
        if (specs->empty()) {
          for (auto n = 0u; n < nspec; ++n) {
            specs->push_back(ZERO);
          }
        }
        raise::ErrorIf(specs->size() != nspec,
                       "Drift vector and/or temperature vector length does "
                       "not match number of species",
                       HERE);
      }
      if (densities.empty()) {
        for (auto n = 0u; n < nspec; n += 2) {
          densities.push_back(TWO / static_cast<real_t>(nspec));
        }
      }
      raise::ErrorIf(densities.size() != nspec / 2,
                     "Density vector length must be half of the number of "
                     "species (per each pair of species)",
                     HERE);
    }

    inline void InitPrtls(Domain<S, M>& domain) {
      const auto nspec = domain.species.size();
      for (auto n = 0u; n < nspec; n += 2) {
        const auto drift_1  = prmvec_t { drifts_in_x[n],
                                        drifts_in_y[n],
                                        drifts_in_z[n] };
        const auto drift_2  = prmvec_t { drifts_in_x[n + 1],
                                        drifts_in_y[n + 1],
                                        drifts_in_z[n + 1] };
        const auto injector = arch::experimental::
          UniformInjector<S, M, arch::experimental::Maxwellian, arch::experimental::Maxwellian>(
            arch::experimental::Maxwellian<S, M>(domain.mesh.metric,
                                                 *domain.random_pool,
                                                 temperatures[n],
                                                 drift_1),
            arch::experimental::Maxwellian<S, M>(domain.mesh.metric,
                                                 *domain.random_pool,
                                                 temperatures[n + 1],
                                                 drift_2),
            { n + 1, n + 2 });
        arch::experimental::InjectUniform<S, M, decltype(injector)>(
          params,
          domain,
          injector,
          densities[n / 2]);
      }
    }
  };

} // namespace user

#endif
