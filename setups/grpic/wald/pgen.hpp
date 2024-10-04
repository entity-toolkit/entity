#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <class M, Dimension D>
  struct InitFields {
    InitFields(M metric_) : metric { metric_ } {}

    Inline auto VerticalPotential(const coord_t<D>& x_Cd) const -> real_t {
      coord_t<D> x_Ph { ZERO };
      metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
      return HALF * SQR(x_Ph[0]) * SQR(math::sin(x_Ph[1]));
    }

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      coord_t<D> xi {ZERO}, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0];
      x0m[1] = xi[1] - HALF;
      x0p[0] = xi[0];
      x0p[1] = xi[1] + HALF;
      
      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
      return (VerticalPotential(x0p) - VerticalPotential(x0m)) * inv_sqrt_detH_ijP;
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      coord_t<D> xi {ZERO}, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0] + HALF - HALF;
      x0m[1] = xi[1];
      x0p[0] = xi[0] + HALF + HALF;
      x0p[1] = xi[1];

      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
      return -(VerticalPotential(x0p) - VerticalPotential(x0m)) * inv_sqrt_detH_ijP;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto dx1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto dx2(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto dx3(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

  private:
    const M metric;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::GRPIC>::value };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Kerr_Schild, Metric::QKerr_Schild, Metric::Kerr_Schild_0>::value
    };
    static constexpr auto dimensions { traits::compatible_with<Dim::_2D>::value };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    InitFields<M, D> init_flds;

    inline PGen(SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , init_flds { m.mesh().metric } {}

    inline PGen() {}
  };

} // namespace user

#endif
