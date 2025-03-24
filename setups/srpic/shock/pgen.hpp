#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

#include <utility>

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    /*
      Sets up magnetic and electric field components for the simulation.
      Must satisfy E = -v x B for Lorentz Force to be zero.

      @param bmag: magnetic field scaling
      @param btheta: magnetic field polar angle
      @param bphi: magnetic field azimuthal angle
      @param drift_ux: drift velocity in the x direction
    */
    InitFields(real_t bmag, real_t btheta, real_t bphi, real_t drift_ux)
      : Bmag { bmag }
      , Btheta { btheta * static_cast<real_t>(convert::deg2rad) }
      , Bphi { bphi * static_cast<real_t>(convert::deg2rad) }
      , Vx { drift_ux } {}

    // magnetic field components
    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return Bmag * math::cos(Btheta);
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return Bmag * math::sin(Btheta) * math::sin(Bphi);
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return Bmag * math::sin(Btheta) * math::cos(Bphi);
    }

    // electric field components
    Inline auto ex1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return -Vx * Bmag * math::sin(Btheta) * math::cos(Bphi);
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return Vx * Bmag * math::sin(Btheta) * math::sin(Bphi);
    }

  private:
    const real_t Btheta, Bphi, Vx, Bmag;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics { traits::compatible_with<Metric::Minkowski>::value };
    static constexpr auto dimensions {
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
    };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    // gas properties
    const real_t drift_ux, temperature, filling_fraction;
    // injector properties
    const real_t injector_velocity, injection_start, dt;
    // magnetic field properties
    real_t        Btheta, Bphi, Bmag;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M> { p }
      , drift_ux { p.template get<real_t>("setup.drift_ux") }
      , temperature { p.template get<real_t>("setup.temperature") }
      , Bmag { p.template get<real_t>("setup.Bmag", ZERO) }
      , Btheta { p.template get<real_t>("setup.Btheta", ZERO) }
      , Bphi { p.template get<real_t>("setup.Bphi", ZERO) }
      , init_flds { Bmag, Btheta, Bphi, drift_ux }
      , filling_fraction { p.template get<real_t>("setup.filling_fraction", 1.0) }
      , injector_velocity { p.template get<real_t>("setup.injector_velocity", 1.0) }
      , injection_start { p.template get<real_t>("setup.injection_start", 0.0) }
      , dt { p.template get<real_t>("algorithms.timestep.dt") } {}

    inline PGen() {}

    auto MatchFields(real_t time) const -> InitFields<D> {
      return init_flds;
    }

    inline void InitPrtls(Domain<S, M>& local_domain) {

      // minimum and maximum position of particles
      real_t xg_min = local_domain.mesh.extent(in::x1).first;
      real_t xg_max = local_domain.mesh.extent(in::x1).first +
                      filling_fraction * (local_domain.mesh.extent(in::x1).second -
                                          local_domain.mesh.extent(in::x1).first);

      // define box to inject into
      boundaries_t<real_t> box;
      // loop over all dimensions
      for (unsigned short d { 0 }; d < static_cast<unsigned short>(M::Dim); ++d) {
        // compute the range for the x-direction
        if (d == static_cast<unsigned short>(in::x1)) {
          box.push_back({ xg_min, xg_max });
        } else {
          // inject into full range in other directions
          box.push_back(Range::All);
        }
      }

      // spatial distribution of the particles
      // -> hack to use the uniform distribution in NonUniformInjector
      const auto spatial_dist = arch::Piston<S, M>(local_domain.mesh.metric,
                                                   xg_min,
                                                   xg_max,
                                                   in::x1);

      // energy distribution of the particles
      const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                      local_domain.random_pool,
                                                      temperature,
                                                      -drift_ux,
                                                      in::x1);

      const auto injector = arch::NonUniformInjector<S, M, arch::Maxwellian, arch::Piston>(
        energy_dist,
        spatial_dist,
        { 1, 2 });

      arch::InjectNonUniform<S, M, arch::NonUniformInjector<S, M, arch::Maxwellian, arch::Piston>>(
        params,
        local_domain,
        injector,
        1.0,   // target density
        false, // no weights
        box);
    }

    void CustomPostStep(std::size_t, long double time, Domain<S, M>& domain) {

      // same maxwell distribution as above
      const auto energy_dist = arch::Maxwellian<S, M>(domain.mesh.metric,
                                                      domain.random_pool,
                                                      temperature,
                                                      -drift_ux,
                                                      in::x1);

      // initial position of injector
      const auto x_init = domain.mesh.extent(in::x1).first +
                          filling_fraction * (domain.mesh.extent(in::x1).second -
                                              domain.mesh.extent(in::x1).first);

      // check if injector is supposed to start moving already
      const auto dt_inj = time - injection_start > ZERO ? 
                          time - injection_start : ZERO;

      // define box to inject into
      boundaries_t<real_t> box;

      // loop over all dimension
      for (auto d = 0u; d < M::Dim; ++d) {
        if (d == 0) {
          box.push_back({ x_init + injector_velocity * dt_inj - 
                          drift_ux / math::sqrt(1 + SQR(drift_ux)) * dt,
                          x_init + injector_velocity * (dt_inj + dt) });
        } else {
          box.push_back(Range::All);
        }
      }

      // spatial distribution of the particles
      // -> hack to use the uniform distribution in NonUniformInjector
      const auto spatial_dist = arch::Piston<S, M>(domain.mesh.metric,
                                                   box[0].first,
                                                   box[0].second,
                                                   in::x1);

      // ToDo: extend Replenish to replace the current injector
      const auto injector = arch::NonUniformInjector<S, M, arch::Maxwellian, arch::Piston>(
        energy_dist,
        spatial_dist,
        { 1, 2 });

      // inject non-uniformly within the defined box
      arch::InjectNonUniform<S, M, decltype(injector)>(params,
                                                       domain,
                                                       injector,
                                                       ONE,
                                                       false,
                                                       box);

      /*
          I thought this option would be better, but I can't get it to work
      */

      //   const auto spatial_dist = arch::Replenish<S, M,
      //   decltype(TargetDensityProfile)>(domain.mesh.metric,
      //                                                   domain.fields.bckp,
      //                                                   box,
      //                                                   TargetDensity,
      //                                                   1.0);

      //   const auto injector = arch::NonUniformInjector<S, M, arch::Maxwellian, arch::Replenish>(
      //       energy_dist,
      //       spatial_dist,
      //       {1, 2});

      // const auto injector = arch::MovingInjector<S, M, in::x1> {
      //   domain.mesh.metric,
      //   domain.fields.bckp,
      //   energy_dist,
      //   box[0].first,
      //   box[0].second,
      //   1.0,
      //   { 1, 2 }
      // };
    }
  };

} // namespace user

#endif
