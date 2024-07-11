#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t bsurf) : Bsurf { bsurf } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return ONE;
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

  private:
    const real_t Bsurf;

  };

  template <Dimension D>
  struct DriveFields : public InitFields<D> {
    DriveFields(real_t time, real_t bsurf, real_t omega)
      : InitFields<D> { bsurf }
      , time { time }
      , Omega { omega } {}

    using InitFields<D>::bx1;
    using InitFields<D>::bx2;
    using InitFields<D>::bx3;

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      return 2.*Omega*x_Ph[1]*(1 + math::tanh(4.*(-1 + time)))*(0.5 - 0.5*math::tanh(7.5*(-0.4 + math::abs(x_Ph[1]))));
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t time, Omega;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Minkowski>::value
    };
    static constexpr auto dimensions { traits::compatible_with<Dim::_2D>::value };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;


    const real_t  Bsurf, Omega;
    const Metadomain<S, M>& global_domain;
    InitFields<D> init_flds;
    
    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , global_domain { m }
      , Omega { p.template get<real_t>("setup.omega") }
      , Bsurf { p.template get<real_t>("setup.Bsurf", ONE) }
      , init_flds {Bsurf} {
      }

    inline PGen() {}

    auto FieldDriver(real_t time) const -> DriveFields<D> {
      const real_t omega_t = Omega;
      return DriveFields<D> { time, Bsurf, omega_t };
    }
  
  };

} // namespace user

#endif
