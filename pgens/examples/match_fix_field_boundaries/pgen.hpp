#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct ZeroFields {
    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions { ::traits::pgen::compatible_with<Dim::_1D> {} };

    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t amplitude, omega;
    const real_t t_transition, t_duration;

    PGen(const SimulationParams& p, const Metadomain<S, M>&)
      : arch::ProblemGenerator<S, M> { p }
      , amplitude { p.template get<real_t>("setup.amplitude", ONE) }
      , omega { p.template get<real_t>("setup.omega", ONE) }
      , t_transition { p.template get<real_t>("setup.t_transition") }
      , t_duration { p.template get<real_t>("setup.t_duration") } {}

    /**
     * Sets up the driving field on the left boundary.
     *
     * @param bc_in Direction of the boundary (only be used for one side, so no need to check)
     * @param comp Electromagnetic component to set
     *
     * @note Because the wave is only set on the boundary, no coordinate dependency is needed.
     *
     * @note The fields are normalized to B0 (nominal magnetic field)
     * @note Launching an Ez x By wave from the left boundary
     * @note else-statement will only be hit if right boundary is also FIXED
     *
     * @return Pair of (value to set, whether to set it or not)
     */
    auto FixFieldsConst(simtime_t time, const bc_in& bc, em comp) const
      -> std::pair<real_t, bool> {
      if (bc == bc_in::Mx1) {
        const auto phase { time * omega };
        real_t     ampl { ZERO };
        if (time < t_transition) {
          ampl = (time / t_transition);
        } else if (time < t_transition + t_duration) {
          ampl = ONE;
        } else {
          ampl = math::max(
            ONE - (static_cast<real_t>(time) - t_transition - t_duration) /
                    t_transition,
            ZERO);
        }
        ampl *= amplitude;

        if (comp == em::ex3) {
          return { -ampl * math::cos(phase), true };
        } else if (comp == em::bx2) {
          return { math::sin(phase) * ampl, true };
        } else {
          return { ZERO, true };
        }
      } else {
        return { ZERO, true };
      }
    }

    /*
     * Enough to only enforce Ey, Ez and Bx to zero on the right boundary
     *
     * @note Only called if one of the boundaries is MATCH
     */
    auto MatchFields(simtime_t) const -> ZeroFields<D> {
      return ZeroFields<D> {};
    }
  };

} // namespace user

#endif
