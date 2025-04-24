#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"

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
    static constexpr auto engines {
      traits::compatible_with<SimEngine::SRPIC, SimEngine::GRPIC>::value
    };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Minkowski,
                              Metric::Spherical,
                              Metric::QSpherical,
                              Metric::Kerr_Schild,
                              Metric::QKerr_Schild,
                              Metric::Kerr_Schild_0>::value
    };
    static constexpr auto dimensions {
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
    };

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
      const auto x1_e  = params.template get<std::vector<real_t>>("setup.x1_e",
                                                                 empty);
      const auto x2_e  = params.template get<std::vector<real_t>>("setup.x2_e",
                                                                 empty);
      const auto x3_e  = params.template get<std::vector<real_t>>("setup.x3_e",
                                                                 empty);
      const auto phi_e = params.template get<std::vector<real_t>>("setup.phi_e",
                                                                  empty);
      const auto ux1_e = params.template get<std::vector<real_t>>("setup.ux1_e",
                                                                  empty);
      const auto ux2_e = params.template get<std::vector<real_t>>("setup.ux2_e",
                                                                  empty);
      const auto ux3_e = params.template get<std::vector<real_t>>("setup.ux3_e",
                                                                  empty);

      const auto x1_i  = params.template get<std::vector<real_t>>("setup.x1_i",
                                                                 empty);
      const auto x2_i  = params.template get<std::vector<real_t>>("setup.x2_i",
                                                                 empty);
      const auto x3_i  = params.template get<std::vector<real_t>>("setup.x3_i",
                                                                 empty);
      const auto phi_i = params.template get<std::vector<real_t>>("setup.phi_i",
                                                                  empty);
      const auto ux1_i = params.template get<std::vector<real_t>>("setup.ux1_i",
                                                                  empty);
      const auto ux2_i = params.template get<std::vector<real_t>>("setup.ux2_i",
                                                                  empty);
      const auto ux3_i = params.template get<std::vector<real_t>>("setup.ux3_i",
                                                                  empty);
      std::map<std::string, std::vector<real_t>> data_e {
        {  "x1",  x1_e },
        {  "x2",  x2_e },
        { "ux1", ux1_e },
        { "ux2", ux2_e },
        { "ux3", ux3_e }
      };
      std::map<std::string, std::vector<real_t>> data_i {
        {  "x1",  x1_i },
        {  "x2",  x2_i },
        { "ux1", ux1_i },
        { "ux2", ux2_i },
        { "ux3", ux3_i }
      };
      if constexpr (M::CoordType == Coord::Cart or D == Dim::_3D) {
        data_e["x3"] = x3_e;
        data_i["x3"] = x3_i;
      } else if constexpr (D == Dim::_2D) {
        data_e["phi"] = phi_e;
        data_i["phi"] = phi_i;
      }

      arch::InjectGlobally<S, M>(global_domain, local_domain, (spidx_t)1, data_e);
      arch::InjectGlobally<S, M>(global_domain, local_domain, (spidx_t)2, data_i);
    }

    auto FixFieldsConst(const bc_in&, const em&) const -> std::pair<real_t, bool> {
      return { ZERO, false };
    }
  };

} // namespace user

#endif
