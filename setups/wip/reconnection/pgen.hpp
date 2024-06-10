#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t Bmag, real_t width, real_t y0)
      : Bmag { Bmag }
      , width { width }
      , y0 { y0 } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return Bmag * math::tanh((x_Ph[1] - y0) / width);
    }

  private:
    const real_t Bmag, width, y0;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics { traits::compatible_with<Metric::Minkowski>::value };
    static constexpr auto dimensions {
      traits::compatible_with<Dim::_2D, Dim::_3D>::value
    };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t  Bmag, width, y0;
    const real_t  inj_pad, y_min, y_max;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , Bmag { p.template get<real_t>("setup.Bmag", 1.0) }
      , width { p.template get<real_t>("setup.width") }
      , y0 { (m.mesh().extent(in::x2).first + m.mesh().extent(in::x2).second) * HALF }
      , inj_pad { p.template get<real_t>("setup.inj_pad") }
      , y_min { m.mesh().extent(in::x2).first }
      , y_max { m.mesh().extent(in::x2).second }
      , init_flds { Bmag, width, y0 } {}

    inline PGen() {}

    template <typename T>
    auto CustomFields(dir::direction_t<M::Dim> direction, real_t time) const
      -> std::pair<std::pair<real_t, real_t>, T> {
      const auto dim  = direction.get_dim();
      const auto sign = direction.get_sign();
      raise::ErrorIf(dim != in::x2, "Custom boundaries only work in +/-y", HERE);
      std::pair<real_t, real_t> range { ZERO, ZERO };
      if (sign < 0) { // -y
        range = { Range::Min, y_min + inj_pad };
      } else { // +y
        range = { y_max - inj_pad, Range::Max };
      }
    }
  };

} // namespace user

#endif
