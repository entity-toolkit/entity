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

    const Metadomain<S, M>& global_domain;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain } {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto empty = std::vector<real_t> {};
      const auto x1s = params.template get<std::vector<real_t>>("setup.x1s", empty);
      const auto y1s = params.template get<std::vector<real_t>>("setup.y1s", empty);
      const auto z1s = params.template get<std::vector<real_t>>("setup.z1s", empty);
      const auto ux1s = params.template get<std::vector<real_t>>("setup.ux1s",
                                                                 empty);
      const auto uy1s = params.template get<std::vector<real_t>>("setup.uy1s",
                                                                 empty);
      const auto uz1s = params.template get<std::vector<real_t>>("setup.uz1s",
                                                                 empty);

      const auto x2s = params.template get<std::vector<real_t>>("setup.x2s", empty);
      const auto y2s = params.template get<std::vector<real_t>>("setup.y2s", empty);
      const auto z2s = params.template get<std::vector<real_t>>("setup.z2s", empty);
      const auto ux2s = params.template get<std::vector<real_t>>("setup.ux2s",
                                                                 empty);
      const auto uy2s = params.template get<std::vector<real_t>>("setup.uy2s",
                                                                 empty);
      const auto uz2s = params.template get<std::vector<real_t>>("setup.uz2s",
                                                                 empty);
      // std::vector<real_t> x, y, z, ux_1, uy_1, uz_1, ux_2, uy_2, uz_2;
      // x.push_back(0.85);
      // x.push_back(0.123);
      // if constexpr (D == Dim::_2D || D == Dim::_3D) {
      //   y.push_back(0.32);
      //   y.push_back(0.321);
      // }
      // if constexpr (D == Dim::_3D) {
      //   z.push_back(0.231);
      //   z.push_back(0.687);
      // }
      // ux_1.push_back(1.0);
      // uy_1.push_back(-1.0);
      // uz_1.push_back(0.0);
      // ux_1.push_back(1.0);
      // uy_1.push_back(-2.0);
      // uz_1.push_back(1.0);
      //
      // ux_2.push_back(1.0);
      // uy_2.push_back(1.0);
      // uz_2.push_back(0.0);
      // ux_2.push_back(-2.0);
      // uy_2.push_back(3.0);
      // uz_2.push_back(-1.0);
      //
      const std::map<std::string, std::vector<real_t>> data_1 {
        { "x1",  x1s},
        { "x2",  y1s},
        { "x3",  z1s},
        {"ux1", ux1s},
        {"ux2", uy1s},
        {"ux3", uz1s}
      };
      const std::map<std::string, std::vector<real_t>> data_2 {
        { "x1",  x2s},
        { "x2",  y2s},
        { "x3",  z2s},
        {"ux1", ux2s},
        {"ux2", uy2s},
        {"ux3", uz2s}
      };

      arch::InjectGlobally<S, M>(global_domain, local_domain, (arch::spidx_t)1, data_1);
      arch::InjectGlobally<S, M>(global_domain, local_domain, (arch::spidx_t)2, data_2);
    }

    // void CustomPostStep(std::size_t, long double time, Domain<S, M>& domain) {
    //   if (time >= 0.1) {
    //     for (auto& species : domain.species) {
    //       auto ux1 = species.ux1;
    //       auto ux2 = species.ux2;
    //       auto ux3 = species.ux3;
    //       Kokkos::parallel_for(
    //         "Stop",
    //         species.rangeActiveParticles(),
    //         Lambda(index_t p) {
    //           ux1(p) = ZERO;
    //           ux2(p) = ZERO;
    //           ux3(p) = ZERO;
    //         });
    //     }
    //   }
    // }
  };

} // namespace user

#endif
