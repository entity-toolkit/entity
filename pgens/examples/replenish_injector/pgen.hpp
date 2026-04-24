#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/spatial_dist.h"
#include "framework/domain/metadomain.h"
#include "kernels/particle_moments.hpp"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct EMFields {
    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return 0.8;
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ONE;
    }
  };

  template <Dimension D>
  struct NonUniformTargetDensity {
    Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
      // example of a non-uniform target density that peaks at the center of the domain
      real_t r2 { ZERO };
      for (auto d = 0u; d < D; ++d) {
        r2 += SQR(x_Ph[d] - HALF);
      }
      return std::exp(-r2 / SQR(static_cast<real_t>(0.2)));
      //                                                ^
      //                        characteristic width of the density profile
    }
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_2D, Dim::_3D> {}
    };

    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    EMFields<D> init_flds;

    const std::string target_density;

    PGen(const SimulationParams& p, const Metadomain<S, M>& /*metadomain*/)
      : arch::ProblemGenerator<S, M> { p }
      , target_density { params.template get<std::string>(
          "setup.target_density") } {}

    void CustomPostStep(timestep_t step, simtime_t /*time*/, Domain<S, M>& domain) {
      if (step % 100u != 0u) {
        return;
      }
      // perform replenishment and injection every 100 timesteps

      { // compute density of species #1 and #2

        //   saves the density to domain.fields.buff(:,:,:,0)
        const auto ni2    = domain.mesh.n_active(in::x2);
        const auto inv_n0 = ONE / params.template get<real_t>("scales.n0");

        auto scatter_buff = Kokkos::Experimental::create_scatter_view(
          domain.fields.buff);
        Kokkos::deep_copy(domain.fields.buff, ZERO);
        for (const auto sp : std::vector<spidx_t> { 1, 2 }) {
          const auto& prtl_spec = domain.species[sp - 1];
          Kokkos::parallel_for("ComputeDensity",
                               prtl_spec.rangeActiveParticles(),
                               kernel::ParticleMoments_kernel<S, M, FldsID::N, 3>(
                                 {},
                                 scatter_buff,
                                 0u,
                                 prtl_spec.i1,
                                 prtl_spec.i2,
                                 prtl_spec.i3,
                                 prtl_spec.dx1,
                                 prtl_spec.dx2,
                                 prtl_spec.dx3,
                                 prtl_spec.ux1,
                                 prtl_spec.ux2,
                                 prtl_spec.ux3,
                                 prtl_spec.phi,
                                 prtl_spec.weight,
                                 prtl_spec.tag,
                                 prtl_spec.mass(),
                                 prtl_spec.charge(),
                                 false,
                                 domain.mesh.metric,
                                 domain.mesh.flds_bc(),
                                 ni2,
                                 inv_n0,
                                 0u));
        }
        Kokkos::Experimental::contribute(domain.fields.buff, scatter_buff);
      }

      const auto energy_dist = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(),
        0.2); // <-- target temperature for injection
      if (target_density == "uniform") {
        // pass the computed density to the replenisher
        const auto replenish_sdist = arch::ReplenishUniform<S, M, 3>(
          domain.mesh.metric,
          domain.fields.buff,
          0u,   // <-- index in buff where the density is stored
          ONE); // <-- target density for replenishment
        arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(replenish_sdist)>(
          params,
          domain,
          { 1, 2 },
          { energy_dist, energy_dist },
          replenish_sdist,
          ONE);
      } else {
        const auto target_density_profile = NonUniformTargetDensity<D> {};
        const auto replenish_sdist =
          arch::Replenish<S, M, 3, decltype(target_density_profile)>(
            domain.mesh.metric,
            domain.fields.buff,
            0u, // <-- index in buff where the density is stored
            target_density_profile,
            ONE); // <-- target density for replenishment
        arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(replenish_sdist)>(
          params,
          domain,
          { 1, 2 },
          { energy_dist, energy_dist },
          replenish_sdist,
          ONE);
      }
    }
  };

} // namespace user

#endif
