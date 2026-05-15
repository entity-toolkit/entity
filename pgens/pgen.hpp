#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"
#include "utils/formatting.h"

#include "framework/domain/metadomain.h"

#include <plog/Log.h>

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct PGen {
    // compatibility traits for the problem generator
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC, SimEngine::GRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski,
                                      Metric::Spherical,
                                      Metric::QSpherical,
                                      Metric::Kerr_Schild,
                                      Metric::QKerr_Schild,
                                      Metric::Kerr_Schild_0> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D> {}
    };

    PGen(const SimulationParams&, const Metadomain<S, M>&) {
      const auto message = fmt::format(
        "Problem generator initialized with `%s` engine and `%dD %s` metric",
        SimEngine(S).to_string(),
        M::Dim,
        Metric(M::MetricType).to_string());
      PLOGI << message;
    }
  };

} // namespace user

#endif
