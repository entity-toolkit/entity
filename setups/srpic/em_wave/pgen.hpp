#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/comparators.h"
#include "utils/numeric.h"
#include "archetypes/particle_injector.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t a, real_t sx1, real_t sx2, real_t sx3, int k1, int k2, int k3, real_t b_bg)
      : amplitude { a }
      , kx1 { (sx1 > ZERO) ? (real_t)(constant::TWO_PI) * (real_t)k1 / sx1 : ZERO }
      , kx2 { (sx2 > ZERO) ? (real_t)(constant::TWO_PI) * (real_t)k2 / sx2 : ZERO }
      , kx3 { (sx3 > ZERO) ? (real_t)(constant::TWO_PI) * (real_t)k3 / sx3 : ZERO }
      , b_bg { b_bg }
      , kmag13 { math::sqrt(SQR(kx1) + SQR(kx3)) }
      , kmag { math::sqrt(SQR(kx1) + SQR(kx2) + SQR(kx3)) } {
      raise::ErrorIf(cmp::AlmostZero_host(kx1) and cmp::AlmostZero_host(kx3),
                     "kx1 and kx3 cannot be zero",
                     HERE);
    }

    // B is in k x y
    // E is in -k x B

    Inline auto arg(const coord_t<D>& x_Ph) const -> real_t {
      if constexpr (D == Dim::_1D) {
        return kx1 * x_Ph[0];
      } else if constexpr (D == Dim::_2D) {
        return kx1 * x_Ph[0] + kx2 * x_Ph[1];
      } else {
        return kx1 * x_Ph[0] + kx2 * x_Ph[1] + kx3 * x_Ph[2];
      }
    }

Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
  // E_x = E0 * (kx2 / |k|) * sin(k · x)
  return amplitude * kx2 / kmag * math::sin(arg(x_Ph));
}

Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
  // E_y = -E0 * (kx1 / |k|) * sin(k · x)
  return -amplitude * kx1 / kmag * math::sin(arg(x_Ph));
}

Inline auto ex3(const coord_t<D>& /*x_Ph*/) const -> real_t {
  return 0.0;
}

Inline auto bx1(const coord_t<D>& /*x_Ph*/) const -> real_t {
  return 0.0;
}

Inline auto bx2(const coord_t<D>& /*x_Ph*/) const -> real_t {
  return 0.0;
}

Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
  // B_z = background + perturbation
  return b_bg + amplitude * math::sin(arg(x_Ph));
}

  private:
    const real_t amplitude;
    const real_t kx1, kx2, kx3, kmag13, kmag, b_bg;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions =
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t  amplitude, bbg;
    const int     kx1, kx2, kx3;
    const real_t  sx1, sx2, sx3;
    InitFields<D> init_flds;
    const Metadomain<S, M>& global_domain;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , amplitude { params.template get<real_t>("setup.amplitude", 1.0) }
      , kx1 { params.template get<int>("setup.kx1", 1) }
      , kx2 { params.template get<int>("setup.kx2", 0) }
      , kx3 { params.template get<int>("setup.kx3", 0) }
      , bbg { params.template get<real_t>("setup.bbg", 0.0) }
      , sx1 { global_domain.mesh().extent(in::x1).second -
              global_domain.mesh().extent(in::x1).first }
      , sx2 { global_domain.mesh().extent(in::x2).second -
              global_domain.mesh().extent(in::x2).first }
      , sx3 { global_domain.mesh().extent(in::x3).second -
              global_domain.mesh().extent(in::x3).first }
      , init_flds { amplitude, sx1, sx2, sx3, kx1, kx2, kx3, bbg }
      , global_domain { global_domain }  {}

  inline void InitPrtls(Domain<S, M>& domain) {

      const auto empty = std::vector<real_t> {};
      const auto x1  = params.template get<std::vector<real_t>>("setup.x1", empty);
      const auto x2  = params.template get<std::vector<real_t>>("setup.x2", empty);
      const auto x3  = params.template get<std::vector<real_t>>("setup.x3", empty);
      const auto ux1 = params.template get<std::vector<real_t>>("setup.ux1", empty);
      const auto ux2 = params.template get<std::vector<real_t>>("setup.ux2", empty);
      const auto ux3 = params.template get<std::vector<real_t>>("setup.ux3", empty);

      std::map<std::string, std::vector<real_t>> data_p {
        {  "x1",  x1 },
        {  "x2",  x2 },
        {  "x3",  x3 },
        { "ux1", ux1 },
        { "ux2", ux2 },
        { "ux3", ux3 }
      };

      arch::InjectGlobally<S, M>(global_domain, domain, (spidx_t)1, data_p);

  }
};

} // namespace user

#endif
