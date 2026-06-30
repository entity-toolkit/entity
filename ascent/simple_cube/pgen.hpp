#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"
#include "utils/numeric.h"

#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

namespace user {
  using namespace ntt;

  /*
   * Initial magnetic field on a cubic domain.
   *
   *   B1 = B2 = 0
   *   B3 = B0 * sin(2 pi (x - x_lo) / Lx) * sin(2 pi (y - y_lo) / Ly)
   *
   * The pattern is purely spatial so the cube can be visualised with
   * Ascent's pseudocolor plot from the very first output cycle.
   */
  template <Dimension D>
  struct CubeB3Field {
    CubeB3Field(real_t b0, real_t lx, real_t ly, real_t x0, real_t y0)
      : B0 { b0 }
      , Lx { lx }
      , Ly { ly }
      , x0 { x0 }
      , y0 { y0 } {}

    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      const auto kx = static_cast<real_t>(constant::TWO_PI) / Lx;
      const auto ky = static_cast<real_t>(constant::TWO_PI) / Ly;
      return B0 * math::sin(kx * (x_Ph[0] - x0)) *
             math::sin(ky * (x_Ph[1] - y0));
    }

    const real_t B0, Lx, Ly, x0, y0;
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_3D> {}
    };

    const SimulationParams& params;
    CubeB3Field<D>          init_flds;

    PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : params { p }
      , init_flds {
        params.template get<real_t>("setup.B0", ONE),
        global_domain.mesh().extent(in::x1).second -
          global_domain.mesh().extent(in::x1).first,
        global_domain.mesh().extent(in::x2).second -
          global_domain.mesh().extent(in::x2).first,
        global_domain.mesh().extent(in::x1).first,
        global_domain.mesh().extent(in::x2).first
      } {}
  };

} // namespace user

#endif
