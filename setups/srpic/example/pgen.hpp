#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

#include <vector>

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {

    InitFields(real_t a, real_t sx2, int kx2)
      : amplitude { a }
      , sx2 { sx2 }
      , kx2 { kx2 } {}

    // only set ex2 and bx3

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      return amplitude * math::sin(constant::TWO_PI * (x_Ph[1] / sx2) *
                                   static_cast<real_t>(kx2));
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return -amplitude * math::cos(constant::TWO_PI * (x_Ph[1] / sx2) *
                                    static_cast<real_t>(kx2));
    }

  private:
    const real_t amplitude;
    const real_t sx2;
    const int    kx2;
  };

  template <Dimension D>
  struct ExtForce {
    const std::vector<unsigned short> species { 1, 2 };

    ExtForce() = default;

    Inline auto fx1(const unsigned short& sp,
                    const real_t&         time,
                    const coord_t<D>&     x_Ph) const -> real_t {
      (void)sp;
      (void)time;
      (void)x_Ph;
      return ZERO;
    }

    Inline auto fx2(const unsigned short& sp,
                    const real_t&         time,
                    const coord_t<D>&     x_Ph) const -> real_t {
      (void)sp;
      (void)time;
      (void)x_Ph;
      return ZERO;
    }

    Inline auto fx3(const unsigned short& sp,
                    const real_t&         time,
                    const coord_t<D>&     x_Ph) const -> real_t {
      (void)sp;
      (void)time;
      (void)x_Ph;
      return ZERO;
    }
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions = traits::compatible_with<Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    InitFields<D>        init_flds;
    ExtForce<M::PrtlDim> ext_force;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , init_flds { params.template get<real_t>("setup.amplitude", 1.0),
                    global_domain.mesh().extent(in::x2).second -
                      global_domain.mesh().extent(in::x2).first,
                    params.template get<int>("setup.kx2", 2) }
      , ext_force {} {}
  };

} // namespace user

#endif
