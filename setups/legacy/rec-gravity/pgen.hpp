#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/spatial_dist.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct Gravity {
    const std::vector<spidx_t> species { 1, 2 };

    Gravity(real_t f, real_t tscale, real_t ymid)
      : force { f }
      , tscale { tscale }
      , ymid { ymid } {}

    Inline auto fx1(const spidx_t&, const simtime_t&, const coord_t<D>&) const
      -> real_t {
      return ZERO;
    }

    Inline auto fx2(const spidx_t&, const simtime_t& t, const coord_t<D>& x_Ph) const
      -> real_t {
      const auto sign { (x_Ph[1] < ymid) ? ONE : -ONE };
      const auto t_ { static_cast<real_t>(t) };
      if (t_ > tscale) {
        return sign * force;
      } else {
        return sign * force * (ONE - math::cos(constant::PI * t_ / tscale)) / TWO;
      }
    }

    Inline auto fx3(const spidx_t&, const simtime_t&, const coord_t<D>&) const
      -> real_t {
      return ZERO;
    }

  private:
    const real_t force, tscale, ymid;
  };

  template <SimEngine::type S, class M>
  struct CurrentLayer : public arch::SpatialDistribution<S, M> {
    CurrentLayer(const M& metric, real_t width, real_t yi)
      : arch::SpatialDistribution<S, M> { metric }
      , width { width }
      , yi { yi } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t override {
      return ONE / SQR(math::cosh((x_Ph[1] - yi) / width));
    }

  private:
    const real_t yi, width;
  };

  template <Dimension D>
  struct InitFields {
    InitFields(real_t Bmag, real_t width, real_t y1, real_t y2)
      : Bmag { Bmag }
      , width { width }
      , y1 { y1 }
      , y2 { y2 } {}

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return Bmag * (math::tanh((x_Ph[1] - y1) / width) -
                     math::tanh((x_Ph[1] - y2) / width) - 1);
    }

  private:
    const real_t Bmag, width, y1, y2;
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

    const real_t  Bmag, width, overdensity, y1, y2, bg_temp;
    InitFields<D> init_flds;

    Gravity<D> ext_force;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , Bmag { p.template get<real_t>("setup.Bmag", 1.0) }
      , width { p.template get<real_t>("setup.width") }
      , overdensity { p.template get<real_t>("setup.overdensity") }
      , bg_temp { p.template get<real_t>("setup.bg_temp") }
      , y1 { m.mesh().extent(in::x2).first +
             INV_4 *
               (m.mesh().extent(in::x2).second - m.mesh().extent(in::x2).first) }
      , y2 { m.mesh().extent(in::x2).first +
             3 * INV_4 *
               (m.mesh().extent(in::x2).second - m.mesh().extent(in::x2).first) }
      , init_flds { Bmag, width, y1, y2 }
      , ext_force {
        p.template get<real_t>("setup.fmag", 0.1) *
          p.template get<real_t>("scales.omegaB0"),
        (m.mesh().extent(in::x1).second - m.mesh().extent(in::x1).first),
        INV_2 * (m.mesh().extent(in::x2).second + m.mesh().extent(in::x2).first)
      } {}

    inline PGen() {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      // background
      const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                      local_domain.random_pool,
                                                      bg_temp);
      const auto injector    = arch::UniformInjector<S, M, arch::Maxwellian>(
        energy_dist,
        { 1, 2 });
      arch::InjectUniform<S, M, decltype(injector)>(params,
                                                    local_domain,
                                                    injector,
                                                    ONE);
      // current layers
      const auto sigma = params.template get<real_t>("scales.sigma0");
      const auto c_omp = params.template get<real_t>("scales.skindepth0");
      const auto cs_drift_beta = math::sqrt(sigma) * c_omp / (width * overdensity);
      const auto cs_drift_gamma = ONE / math::sqrt(ONE - SQR(cs_drift_beta));
      const auto cs_drift_u     = cs_drift_beta * cs_drift_gamma;
      const auto cs_temp        = HALF * sigma / overdensity;

      for (auto i = 0; i < 2; ++i) {
        const auto drift_vel = (i == 0) ? cs_drift_u : -cs_drift_u;
        const auto y_cs      = (i == 0) ? y1 : y2;
        auto       edist_cs  = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                               local_domain.random_pool,
                                               cs_temp,
                                               drift_vel,
                                               in::x1,
                                               false);
        const auto sdist_cs  = CurrentLayer<S, M>(local_domain.mesh.metric,
                                                 width,
                                                 y_cs);
        const auto inj_cs = arch::NonUniformInjector<S, M, arch::Maxwellian, CurrentLayer>(
          edist_cs,
          sdist_cs,
          { 1, 2 });
        arch::InjectNonUniform<S, M, decltype(inj_cs)>(params,
                                                       local_domain,
                                                       inj_cs,
                                                       overdensity);
      }
    } // namespace user
  };

} // namespace user

#endif
