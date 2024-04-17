#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include "archetypes/field_setter.h"
#include "archetypes/problem_generator.h"

namespace user {
  using namespace ntt;

  // template <Dimension D>
  // struct InitFields {

  //   InitFields(real_t a) : amplitude { a } {}

  //   // only set ex2 and bx3

  //   Inline auto ex2(const coord_t<D>&) const -> real_t {
  //     return value;
  //   }

  //   Inline auto bx3(const coord_t<D>&) const -> real_t {
  //     return -value;
  //   }

  // private:
  //   const real_t amplitude;
  //   const real_t sx2;
  //   const int    kx2;
  // };

  template <SimEngine::type S, class M>
  struct PGen : public ProblemGenerator<S, M> {
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
    using ProblemGenerator<S, M>::D;
    using ProblemGenerator<S, M>::C;
    using ProblemGenerator<S, M>::params;

    inline PGen(SimulationParams& p) : ProblemGenerator<S, M>(p) {
      const auto message = fmt::format(
        "Problem generator initialized with `%s` engine and `%dD %s` metric",
        SimEngine(S).to_string(),
        D,
        Metric(M::MetricType).to_string());
      info::Print(message);
    }
  };

} // namespace user

#endif
