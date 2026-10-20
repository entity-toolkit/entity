#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"

#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

namespace user {

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };

    static constexpr auto engines =
      ::traits::pgen::compatible_with<SimEngine::HYBRID> {};
    static constexpr auto metrics =
      ::traits::pgen::compatible_with<Metric::Minkowski> {};
    static constexpr auto dimensions =
      ::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D> {};

    PGen(const SimulationParams&, const Metadomain<S, M>&) {}
  };

} // namespace user

#endif
