#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/spatial_dist.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct GaussianBlob : public arch::SpatialDistribution<S, M> {
    GaussianBlob(const M& metric, const coord_t<M::Dim>& c, real_t s)
      : arch::SpatialDistribution<S, M> { metric }
      , spreadSqr { SQR(s) } {
      for (auto d = 0u; d < M::Dim; ++d) {
        center[d] = c[d];
      }
    }

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t override {
      // assuming spherical
      auto rSqr = SQR(x_Ph[0]) + SQR(center[0]) -
                  TWO * x_Ph[0] * center[0] * math::cos(x_Ph[1] - center[1]);
      return std::exp(-rSqr / (2 * spreadSqr));
    }

  private:
    tuple_t<real_t, M::Dim> center;
    const real_t            spreadSqr;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics =
      traits::compatible_with<Metric::Spherical, Metric::QSpherical>::value;
    static constexpr auto dimensions = traits::compatible_with<Dim::_2D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t r0, theta0, dr;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>&)
      : arch::ProblemGenerator<S, M> { p }
      , r0 { p.template get<real_t>("setup.r0") }
      , theta0 { p.template get<real_t>("setup.theta0") }
      , dr { p.template get<real_t>("setup.dr") } {}

    inline void InitPrtls(Domain<S, M>& domain) {
      const auto injector = arch::NonUniformInjector<S, M, arch::ColdDist, GaussianBlob>(
        arch::ColdDist<S, M>(domain.mesh.metric),
        GaussianBlob<S, M>(domain.mesh.metric, { r0, theta0 }, dr),
        { 1, 2 });
      arch::InjectNonUniform<S, M, decltype(injector)>(params,
                                                       domain,
                                                       injector,
                                                       1.0,
                                                       true);
      const auto& metric = domain.mesh.metric;
      for (auto& species : domain.species) {
        auto i1   = species.i1;
        auto i2   = species.i2;
        auto dx1  = species.dx1;
        auto dx2  = species.dx2;
        auto phi  = species.phi;
        auto pld1 = species.pld[0];
        auto pld2 = species.pld[1];
        auto pld3 = species.pld[2];
        Kokkos::parallel_for(
          "SavePositions",
          species.npart(),
          Lambda(index_t p) {
            const coord_t<M::PrtlDim> x_Code {
              static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p)),
              static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p)),
              phi(p)
            };
            coord_t<M::PrtlDim> x_XYZ { ZERO };

            metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code, x_XYZ);
            pld1(p) = x_XYZ[0];
            pld2(p) = x_XYZ[1];
            pld3(p) = x_XYZ[2];
          });
      }
    }

    void CustomPostStep(std::size_t, long double time, Domain<S, M>& domain) {
      const auto& metric      = domain.mesh.metric;
      auto&       random_pool = domain.random_pool;
      const auto  time_dbl    = static_cast<double>(time);
      for (auto s = 0u; s < domain.species.size(); ++s) {
        auto& species = domain.species[s];
        auto  ux1     = species.ux1;
        auto  ux2     = species.ux2;
        auto  ux3     = species.ux3;
        auto  i1      = species.i1;
        auto  i2      = species.i2;
        auto  dx1     = species.dx1;
        auto  dx2     = species.dx2;
        auto  phi     = species.phi;
        auto  pld1    = species.pld[0];
        auto  pld2    = species.pld[1];
        auto  pld3    = species.pld[2];
        Kokkos::parallel_for(
          "SavePositions",
          species.npart(),
          ClassLambda(index_t p) {
            const coord_t<M::PrtlDim> x_Code {
              static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p)),
              static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p)),
              phi(p)
            };
            coord_t<M::PrtlDim> x_XYZ { ZERO };

            metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code, x_XYZ);
            auto rand_gen = random_pool.get_state();
            if (s == 0) {
              const auto dist = math::sqrt(SQR(x_XYZ[0] - pld1(p)) +
                                           SQR(x_XYZ[1] - pld2(p)) +
                                           SQR(x_XYZ[2] - pld3(p)));
              if (time_dbl < 20.0) {
                ux1(p) = 0.0;
                ux2(p) = 0.0;
                ux3(p) = -0.9;
              } else if (time_dbl < 25.0) {
                ux1(p) = 0.9;
                ux2(p) = 0.0;
                ux3(p) = 0.0;
              } else if (time_dbl < 45.0) {
                ux1(p) = 0.0;
                ux2(p) = 0.0;
                ux3(p) = 0.9;
              } else if (time_dbl < 50.0) {
                ux1(p) = 0.9;
                ux2(p) = 0.0;
                ux3(p) = 0.0;
              } else {
                ux1(p) = 5.0 * (Random<real_t>(rand_gen) - HALF);
                ux2(p) = 5.0 * (Random<real_t>(rand_gen) - HALF);
                ux3(p) = 5.0 * (Random<real_t>(rand_gen) - HALF);
              }
            }
            // add a slight spread
            ux1(p) += 0.002 * (Random<real_t>(rand_gen) - HALF);
            ux2(p) += 0.002 * (Random<real_t>(rand_gen) - HALF);
            ux3(p) += 0.002 * (Random<real_t>(rand_gen) - HALF);
            random_pool.free_state(rand_gen);
          });
      }
    }
  };

} // namespace user

#endif