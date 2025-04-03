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

#include "kernels/particle_moments.hpp"

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct CurrentLayer : public arch::SpatialDistribution<S, M> {
    CurrentLayer(const M& metric, real_t cs_width, real_t cs_y)
      : arch::SpatialDistribution<S, M> { metric }
      , cs_width { cs_width }
      , cs_y { cs_y } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t override {
      return ONE / SQR(math::cosh((x_Ph[1] - cs_y) / cs_width));
    }

  private:
    const real_t cs_y, cs_width;
  };

  // field initializer
  template <Dimension D>
  struct InitFields {
    InitFields(real_t bg_B, real_t bg_Bguide, real_t cs_width, real_t cs_y)
      : bg_B { bg_B }
      , bg_Bguide { bg_Bguide }
      , cs_width { cs_width }
      , cs_y { cs_y } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return bg_B * (math::tanh((x_Ph[1] - cs_y) / cs_width));
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return bg_Bguide;
    }

  private:
    const real_t bg_B, bg_Bguide, cs_width, cs_y;
  };

  template <Dimension D>
  struct BoundaryFieldsInX1 {
    BoundaryFieldsInX1(real_t bg_B,
                       real_t bg_Bguide,
                       real_t beta_rec,
                       real_t cs_width,
                       real_t cs_x,
                       real_t cs_y)
      : bg_B { bg_B }
      , bg_Bguide { bg_Bguide }
      , beta_rec { beta_rec }
      , cs_width { cs_width }
      , cs_x { cs_x }
      , cs_y { cs_y } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return bg_B * (math::tanh((x_Ph[1] - cs_y) / cs_width));
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return beta_rec * bg_B * (math::tanh((x_Ph[0] - cs_x) / cs_width));
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return bg_Bguide;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return beta_rec * bg_Bguide * math::tanh((x_Ph[1] - cs_y) / cs_width);
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return -beta_rec * bg_B;
    }

  private:
    const real_t bg_B, bg_Bguide, beta_rec, cs_width, cs_x, cs_y;
  };

  template <Dimension D>
  struct BoundaryFieldsInX2 {
    BoundaryFieldsInX2(real_t bg_B, real_t bg_Bguide, real_t cs_width, real_t cs_y)
      : bg_B { bg_B }
      , bg_Bguide { bg_Bguide }
      , cs_width { cs_width }
      , cs_y { cs_y } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return bg_B * (math::tanh((x_Ph[1] - cs_y) / cs_width));
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return bg_Bguide;
    }

    Inline auto ex1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t bg_B, bg_Bguide, cs_width, cs_y;
  };

  // constant particle density for particle boundaries
  template <SimEngine::type S, class M>
  struct ConstDens {
    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      return ONE;
    }
  };
  template <SimEngine::type S, class M>
  using spatial_dist_t = arch::Replenish<S, M, ConstDens<S, M>>;

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

    const real_t bg_B, bg_Bguide, bg_temperature;
    const real_t cs_width, cs_overdensity, cs_x, cs_y;

    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , bg_B { p.template get<real_t>("setup.bg_B", 1.0) }
      , bg_Bguide { p.template get<real_t>("setup.bg_Bguide", 0.0) }
      , bg_temperature { p.template get<real_t>("setup.bg_temperature", 0.001) }
      , cs_width { p.template get<real_t>("setup.cs_width") }
      , cs_overdensity { p.template get<real_t>("setup.cs_overdensity") }
      , cs_x { m.mesh().extent(in::x1).first +
               INV_2 * (m.mesh().extent(in::x1).second -
                        m.mesh().extent(in::x1).first) }
      , cs_y { m.mesh().extent(in::x2).first +
               INV_2 * (m.mesh().extent(in::x2).second -
                        m.mesh().extent(in::x2).first) }
      , init_flds { bg_B, bg_Bguide, cs_width, cs_y } {}

    inline PGen() {}

    auto MatchFieldsInX1(simtime_t) const -> BoundaryFieldsInX1<D> {
      return BoundaryFieldsInX1<D> { bg_B,     bg_Bguide, (real_t)0.1,
                                     cs_width, cs_x,      cs_y };
    }

    auto MatchFieldsInX2(simtime_t) const -> BoundaryFieldsInX2<D> {
      return BoundaryFieldsInX2<D> { bg_B, bg_Bguide, cs_width, cs_y };
    }

    inline void InitPrtls(Domain<S, M>& local_domain) {
      // background
      const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                      local_domain.random_pool,
                                                      bg_temperature);
      const auto injector    = arch::UniformInjector<S, M, arch::Maxwellian>(
        energy_dist,
        { 1, 2 });
      arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(
        params,
        local_domain,
        injector,
        ONE);

      const auto sigma = params.template get<real_t>("scales.sigma0");
      const auto c_omp = params.template get<real_t>("scales.skindepth0");
      const auto cs_drift_beta = math::sqrt(sigma) * c_omp /
                                 (cs_width * cs_overdensity);
      const auto cs_drift_gamma = ONE / math::sqrt(ONE - SQR(cs_drift_beta));
      const auto cs_drift_u     = cs_drift_beta * cs_drift_gamma;
      const auto cs_temperature = HALF * sigma / cs_overdensity;

      // current layer
      auto       edist_cs = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                             local_domain.random_pool,
                                             cs_temperature,
                                             cs_drift_u,
                                             in::x3,
                                             false);
      const auto sdist_cs = CurrentLayer<S, M>(local_domain.mesh.metric,
                                               cs_width,
                                               cs_y);
      const auto inj_cs = arch::NonUniformInjector<S, M, arch::Maxwellian, CurrentLayer>(
        edist_cs,
        sdist_cs,
        { 1, 2 });
      arch::InjectNonUniform<S, M, decltype(inj_cs)>(params,
                                                     local_domain,
                                                     inj_cs,
                                                     cs_overdensity);
    }

    void CustomPostStep(std::size_t step, long double time, Domain<S, M>& domain) {
      //       // 0. define target density profile and box where it has to be
      //       reached
      //       // 1. compute density
      //       // 2. define spatial distribution
      //       // 3. define energy distribution
      //       // 4. define particle injector
      //       // 5. inject particles
      //
      //       // step 0.
      //       // defining the regions of interest (lower and upper
      //       y-boundaries) boundaries_t<real_t> box_upper, box_lower; const
      //       auto [xmin, xmax] = domain.mesh.extent(in::x1); const auto [ymin,
      //       ymax] = domain.mesh.extent(in::x2); box_upper.push_back({ xmin,
      //       xmax }); box_lower.push_back({ xmin, xmax });
      //
      //       box_upper.push_back({ ymax - dy, ymax });
      //       box_lower.push_back({ ymin, ymin + dy });
      //
      //       if constexpr (M::Dim == Dim::_3D) {
      //         const auto [zmin, zmax] = domain.mesh.extent(in::x3);
      //         box_upper.push_back({ zmin, zmax });
      //         box_lower.push_back({ zmin, zmax });
      //       }
      //
      //       const auto const_dens = ConstDens<S, M>();
      //       const auto inv_n0     = ONE / n0;
      //
      //       // step 1 compute density
      //       auto scatter_bckp = Kokkos::Experimental::create_scatter_view(
      //         domain.fields.bckp);
      //       for (auto& prtl_spec : domain.species) {
      //         // clang-format off
      //         Kokkos::parallel_for(
      //           "ComputeMoments",
      //           prtl_spec.rangeActiveParticles(),
      //           kernel::ParticleMoments_kernel<S, M, FldsID::N, 6>({},
      //           scatter_bckp, 0,
      //                                                     prtl_spec.i1,
      //                                                     prtl_spec.i2,
      //                                                     prtl_spec.i3, prtl_spec.dx1,
      //                                                     prtl_spec.dx2,
      //                                                     prtl_spec.dx3, prtl_spec.ux1,
      //                                                     prtl_spec.ux2,
      //                                                     prtl_spec.ux3, prtl_spec.phi,
      //                                                     prtl_spec.weight,
      //                                                     prtl_spec.tag, prtl_spec.mass(),
      //                                                     prtl_spec.charge(),
      //                                                     false,
      //                                                     domain.mesh.metric,
      //                                                     domain.mesh.flds_bc(),
      //                                                     0, inv_n0, 0));
      // 	}
      // 	Kokkos::Experimental::contribute(domain.fields.bckp, scatter_bckp);
      //
      //
      // 	//step 2. define spatial distribution
      // //	const auto spatial_dist = arch::Replenish<S, M,
      // user::ConstDens<S,M>>(domain.mesh.metric, domain.fields.bckp, 0,
      // const_dens, ONE);
      //   const auto spatial_dist = spatial_dist_t<S,M>(domain.mesh.metric,
      //   domain.fields.bckp,0,const_dens,ONE);
      // 	//step 3. define energy distribution
      // 	const auto energy_dist = arch::Maxwellian<S, M>(domain.mesh.metric,
      // domain.random_pool, up_temperature);
      // 	//step 4. define particle injector
      // 	const auto injector = arch::NonUniformInjector<S, M, arch::Maxwellian,
      // spatial_dist_t>(energy_dist, spatial_dist, {1,2});
      // 	//step 5. inject particles
      // 	arch::InjectNonUniform<S, M, decltype(injector)>(params, domain,
      // injector, ONE, false, box_upper); //upper boudary arch::InjectNonUniform<S,
      // M, decltype(injector)>(params, domain, injector, ONE, false,
      // box_lower); //lower boundary
      //
      //
    }
  };

} // namespace user

#endif
