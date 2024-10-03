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

  template <Dimension D>
  struct InitFields {
    InitFields() {}

    Inline auto VerticalPotential(const coord_t<D>& x_Ph) const -> real_t {
      return HALF * SQR(x_Ph[0]) * SQR(math::sin(x_Ph[1]));
    }

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return x_Ph[0] * math::cos(x_Ph[1]);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return -x_Ph[0] * math::sin(x_Ph[1]);
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
    // const real_t Bsurf, Rstar;
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
  
    InitFields<D> init_flds;

    inline PGen(SimulationParams& p, const Metadomain<S, M>& m) :
      arch::ProblemGenerator<S, M>(p) 
      // , init_flds {  } 
      {}

    inline PGen() {}
  };


} // namespace user

#endif
