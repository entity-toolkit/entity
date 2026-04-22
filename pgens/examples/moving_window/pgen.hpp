#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"

#include "archetypes/moving_window.h"
#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/spatial_dist.h"
#include "framework/domain/metadomain.h"
#include "kernels/particle_moments.hpp"

/* -------------------------------------------------------------------------- */
/* Local macros    (same as in particle_pusher_sr.hpp)                        */
/* -------------------------------------------------------------------------- */
#define from_Xi_to_i(XI, I)                                                    \
  {                                                                            \
    I = static_cast<int>((XI + 1)) - 1;                                        \
  }

#define from_Xi_to_i_di(XI, I, DI)                                             \
  {                                                                            \
    from_Xi_to_i((XI), (I));                                                   \
    DI = static_cast<prtldx_t>((XI)) - static_cast<prtldx_t>(I);               \
  }

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))


namespace user {
  using namespace ntt;

  template <Dimension D>
  struct NonUniformTargetDensity {
    Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
      // example of a non-uniform target density that peaks at the center of the domain
      real_t r2 { ZERO };
      for (auto d = 0u; d < D; ++d) {
        r2 += SQR(x_Ph[d] - 0.5);
      }
      return std::exp(
        -r2 / SQR(0.2)); // <-- characteristic width of the density profile
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

    Metadomain<S, M>& global_domain;

    int    i_window, initial_i_window;
    real_t di_window, window_velocity_cd, window_position_cd, window_position;
    const real_t window_velocity;

    const real_t dt;

    inline PGen(const SimulationParams& p, Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain }
      , window_velocity { p.template get<real_t>("setup.window_velocity", 0.0) }
      , dt { p.template get<real_t>("algorithms.timestep.dt") } {}

    inline void InitPrtls(Domain<S, M>& domain) {

      // set up window properties
      window_position = ZERO;
      window_position_cd = domain.mesh.metric.template convert<1, Crd::Ph, Crd::Cd>(
        window_position);
      // convert to cell units and get cell index and sub-cell position
      from_Xi_to_i_di(window_position_cd, i_window, di_window);
      // store initial cell index of piston for window updates
      initial_i_window = i_window;

      // window velocity in code units
      window_velocity_cd = domain.mesh.metric.template transform<1, Idx::XYZ, Idx::U>(
        {},
        window_velocity);
      
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
      // inject particles
      const auto energy_dist = arch::Maxwellian<S, M>(
        domain.mesh.metric,
        domain.random_pool(),
        0.2,
        { window_velocity, ZERO, ZERO });

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

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {
      // update window position
      di_window += window_velocity_cd * dt;
      i_window += static_cast<int>(di_window >= ONE);
      di_window -= (di_window >= ONE);

      // check if the window should be moved
      if ((i_window - initial_i_window) >= N_GHOSTS) {

        // move the window and all fields and particles in it
        arch::MoveWindow<S, M, in::x1>(domain, global_domain, N_GHOSTS);

        // update window index for next update
        i_window -= N_GHOSTS;
      }
    }
  };

} // namespace user

#undef from_Xi_to_i
#undef from_Xi_to_i_di
#undef i_di_to_Xi

#endif
