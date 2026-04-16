#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "archetypes/problem_generator.h"
#include "archetypes/traits.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <SimEngine S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    static constexpr auto engines {
      arch::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      arch::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      arch::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D> {}
    };

    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& metadomain)
      : arch::ProblemGenerator<S, M> { p } {}
  };

} // namespace user

#endif
