#include "enums.h"

#include "arch/traits.h"
#include "traits/pgen.h"
#include "utils/error.h"

#include "framework/simulation.h"
#include "framework/specialization_registry.h"

#include "engines/grpic.hpp"
#include "engines/srpic/srpic.hpp"
#include "pgen.hpp"

#include <type_traits>

namespace ntt {
  template <SimEngine::type S>
  struct EngineSelector;

  template <>
  struct EngineSelector<SimEngine::SRPIC> {
    template <SRMetricClass M>
    using type = SRPICEngine<M>;
  };

  template <>
  struct EngineSelector<SimEngine::GRPIC> {
    template <GRMetricClass M>
    using type = GRPICEngine<M>;
  };
} // namespace ntt

template <auto N, auto... Is>
static constexpr bool is_compatible(::traits::pgen::compatible_with<Is...>) {
  return ((N == Is) || ...);
}

template <ntt::SimEngine::type S, template <Dimension> class M, Dimension D>
static constexpr bool should_compile {
  is_compatible<S>(user::PGen<S, M<D>>::engines) and
  is_compatible<M<D>::MetricType>(user::PGen<S, M<D>>::metrics) and
  is_compatible<D>(user::PGen<S, M<D>>::dimensions)
};

template <ntt::SimEngine::type S, template <Dimension> class M, Dimension D>
void dispatch_engine(ntt::Simulation& sim) {
  if constexpr (S == SimEngine::SRPIC) {
    sim.run<ntt::EngineSelector<S>::template type, M, D>();
  } else if constexpr (S == SimEngine::GRPIC) {
    sim.run<ntt::EngineSelector<S>::template type, M, D>();
  } else {
    static_assert(::traits::always_false<std::integral_constant<SimEngine, S>>::value,
                  "Unsupported engine");
  }
}

auto main(int argc, char* argv[]) -> int {
  ntt::Simulation sim { argc, argv };

  auto matched  = false;
  auto launched = false;

  ntt::for_each_specialization([&](auto spec) {
    using Spec             = decltype(spec);
    const auto requested_e = sim.requested_engine();
    const auto requested_m = sim.requested_metric();
    const auto requested_d = sim.requested_dimension();

    if (requested_e == Spec::engine && requested_m == Spec::metric &&
        requested_d == Spec::dimension) {
      matched = true;
      if constexpr (
        should_compile<Spec::engine, Spec::template MetricTemplateType, Spec::dimension>) {
        dispatch_engine<Spec::engine, Spec::template MetricTemplateType, Spec::dimension>(
          sim);
        launched = true;
      } else {
        raise::Fatal(
          "Requested configuration is not available for this problem generator",
          HERE);
      }
    }
  });

  if (not matched) {
    raise::Fatal("Invalid engine, metric, or dimension combination", HERE);
  }

  if (not launched) {
    raise::Fatal("Requested combination is not enabled in this build", HERE);
  }

  return 0;
}
