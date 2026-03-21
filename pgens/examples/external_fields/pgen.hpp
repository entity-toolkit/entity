#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/traits.h"
#include "framework/domain/metadomain.h"

#include <Kokkos_Pair.hpp>

#include <map>
#include <string>
#include <vector>

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct ExtFields {
    ExtFields(simtime_t time, spidx_t sp) : time { time }, sp { sp } {}

    Inline auto fx1(const coord_t<D>&) const -> real_t {
      return (sp % 2u == 0u) ? -HALF : HALF;
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ONE + 0.2 * time;
    }

    const simtime_t time;
    const spidx_t   sp;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    static constexpr auto engines {
      arch::traits::pgen::compatible_with<SimEngine::SRPIC>::value
    };
    static constexpr auto metrics {
      arch::traits::pgen::compatible_with<Metric::Minkowski>::value
    };
    static constexpr auto dimensions {
      arch::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
    };

    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const Metadomain<S, M>& metadomain;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& metadomain)
      : arch::ProblemGenerator<S, M> { p }
      , metadomain { metadomain } {}

    inline auto ExternalFields(simtime_t           time,
                               spidx_t             sp,
                               const Domain<S, M>& domain) const
      -> std::pair<bool, ExtFields<M::Dim>> {
      // apply only to species 2 and 3
      return {
        sp != 1u,
        ExtFields<M::Dim> { time, sp }
      };
    }

    inline void InitPrtls(Domain<S, M>& domain) {
      const auto prtls = params.template get<std::vector<std::vector<real_t>>>(
        "setup.prtls");
      const auto prtl_species = params.template get<std::vector<int>>(
        "setup.prtl_species");
      raise::ErrorIf(prtl_species.size() != prtls.size(),
                     "setup.prtls_species should be a vector of the same size "
                     "as setup.prtls",
                     HERE);
      if (prtls.size() > 0u) {
        raise::ErrorIf(prtls[0].size() != 3u + static_cast<unsigned int>(D),
                       "setup.prtls should be a vector of vectors of size 3+D",
                       HERE);
        for (auto p = 0u; p < prtls.size(); ++p) {
          const auto prtl      = prtls[p];
          const auto prtl_spec = prtl_species[p];
          std::map<std::string, std::vector<real_t>> data_arr;
          data_arr["x1"] = { prtl[0] };
          if constexpr (D == Dim::_2D or D == Dim::_3D) {
            data_arr["x2"] = { prtl[1] };
          }
          if constexpr (D == Dim::_3D) {
            data_arr["x3"] = { prtl[2] };
          }
          if constexpr (D == Dim::_1D) {
            data_arr["ux1"] = { prtl[1] };
            data_arr["ux2"] = { prtl[2] };
            data_arr["ux3"] = { prtl[3] };
          }
          if constexpr (D == Dim::_2D) {
            data_arr["ux1"] = { prtl[2] };
            data_arr["ux2"] = { prtl[3] };
            data_arr["ux3"] = { prtl[4] };
          }
          if constexpr (D == Dim::_3D) {
            data_arr["ux1"] = { prtl[3] };
            data_arr["ux2"] = { prtl[4] };
            data_arr["ux3"] = { prtl[5] };
          }
          raise::ErrorIf(
            prtl_spec <= 0 or prtl_spec > domain.species.size(),
            "setup.prtl_species should be a vector of integers between 1 and "
            "the number of species in the simulation",
            HERE);
          arch::InjectGlobally<S, M>(metadomain, domain, prtl_spec, data_arr);
        }
      }
    }
  };

} // namespace user

#endif
