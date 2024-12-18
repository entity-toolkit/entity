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

    Inline auto A_3(const coord_t<D>& x_Cd) const -> real_t {
      // return HALF * (metric.template h_<3, 3>(x_Cd)
      //        + TWO * metric.spin() * metric.template h_<1, 3>(x_Cd) * metric.beta1(x_Cd)
      // );
      coord_t<D> x_Ph { ZERO };
      metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
      return HALF * SQR(x_Ph[0]) * SQR(math::sin(x_Ph[1]));
    }

    Inline auto A_1(const coord_t<D>& x_Cd) const -> real_t {
      return HALF * (metric.template h_<1, 3>(x_Cd) 
             + TWO * metric.spin() * metric.template h_<1, 1>(x_Cd) * metric.beta1(x_Cd)
      );
    }

    Inline auto A_0(const coord_t<D>& x_Cd) const -> real_t {
      real_t g_00 { -metric.alpha(x_Cd) * metric.alpha(x_Cd) 
                   + metric.template h_<1, 1>(x_Cd) * metric.beta1(x_Cd) * metric.beta1(x_Cd) 
                  };
      return HALF * (metric.template h_<1, 3>(x_Cd) * metric.beta1(x_Cd) 
             + TWO * metric.spin() * g_00);
    }

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      coord_t<D> xi {ZERO}, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0];
      x0m[1] = xi[1] - HALF;
      x0p[0] = xi[0];
      x0p[1] = xi[1] + HALF;
      
      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };

      if (cmp::AlmostZero(x_Ph[1]))
        return ONE;
      else
        return (A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ijP;
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      coord_t<D> xi {ZERO}, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0] - HALF;
      x0m[1] = xi[1];
      x0p[0] = xi[0] + HALF;
      x0p[1] = xi[1];

      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
      if (cmp::AlmostZero(x_Ph[1]))
        return ZERO;
      else
        return -(A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ijP;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      // coord_t<D> xi {ZERO}, x0m { ZERO }, x0p { ZERO };
      // metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      // x0m[0] = xi[0] + HALF - HALF;
      // x0m[1] = xi[1];
      // x0p[0] = xi[0] + HALF + HALF;
      // x0p[1] = xi[1];

      // real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
      // return -(A_1(x0p) - A_1(x0m)) * inv_sqrt_detH_ijP;
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
      // , xi_min { p.template get<std::vector<real_t>>("setup.inj_rmin") }
      // , xi_max { p.template get<std::vector<real_t>>("setup.inj_rmax") }
      , init_flds { m.mesh().metric } {}
    
    // inline void InitPrtls(Domain<S, M>& local_domain) {
    //     const auto energy_dist = arch::ColdDist<S, M>(local_domain.mesh.metric);
    //     const auto spatial_dist = PointDistribution<S, M>(domain.mesh.metric,
    //                                               xi_min,
    //                                               xi_max);
    //     const auto injector = arch::NonUniformInjector<S, M, arch::ColdDist, PointDistribution>(
    //       energy_dist,
    //       spatial_dist,
    //       { 1, 2 });
    //     arch::InjectNonUniform<S, M, arch::NonUniformInjector<S, M, arch::ColdDist, PointDistribution>>(params,
    //                                                   local_domain,
    //                                                   injector,
    //                                                   1.0);
    // }

    // inline PGen() {}
  };

} // namespace user

#endif
