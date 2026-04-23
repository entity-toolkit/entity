#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"

#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
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

    /*
     * Particle's equation of motion is:
     *
     * du / dt = f_ext + (q / mc) * ((E + E_ext) + v x (B + B_ext))
     *
     * in dimensionless terms:
     *
     * du / dt = f_ext + (q / m) / (q0 / m0) * omegaB0 * ((e + e_ext) + v x (b + b_ext))
     *
     * - f_ext is the external force-field (acceleration) defined here
     * - E and B are interpolated fields from the grid
     * - e = E / B0 and b = B / B0
     * - e_ext = E_ext / B0 and b_ext = B_ext / B0 are the dimensionless external fields defined here
     *
     */

    // f_ext: external force-field (acceleration):
    Inline auto fx1(const coord_t<D>&) const -> real_t {
      return (sp % 2u == 0u) ? -HALF : HALF;
    }

    // b_ext: external magnetic field:
    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ONE + 0.2 * time;
    }

    const simtime_t time;
    const spidx_t   sp;
  };

  template <SimEngine::type S, class M>
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

    PGen(const SimulationParams& p, const Metadomain<S, M>& metadomain)
      : arch::ProblemGenerator<S, M> { p }
      , metadomain { metadomain } {}

    /*
     * @returns a pair of (apply_external_fields, external_fields)
     *
     * @note apply_external_fields is true for species other than 1 (i.e., 2 and 3 in this case)
     */
    auto ExternalFields(simtime_t time, spidx_t sp, const Domain<S, M>& /*domain*/) const
      -> std::pair<bool, ExtFields<M::Dim>> {
      // apply only to species 2 and 3
      return {
        sp != 1u,
        ExtFields<M::Dim> { time, sp }
      };
    }

    void InitPrtls(Domain<S, M>& domain) {
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
