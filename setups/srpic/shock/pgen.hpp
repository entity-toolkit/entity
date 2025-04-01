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
    const int   injection_frequency;
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
      , injection_frequency { p.template get<int>("setup.injection_frequency", 100) }
      , dt { p.template get<real_t>("algorithms.timestep.dt") } {}

    inline PGen() {}

    auto MatchFields(real_t time) const -> InitFields<D> {
      return init_flds;
    }

    auto ResetFields(const em& comp) const -> real_t {
      if (comp == em::ex1) {
        return init_flds.ex1({ ZERO });
      } else if (comp == em::ex2) {
        return init_flds.ex2({ ZERO });
      } else if (comp == em::ex3) {
        return init_flds.ex3({ ZERO });
      } else if (comp == em::bx1) {
        return init_flds.bx1({ ZERO });
      } else if (comp == em::bx2) {
        return init_flds.bx2({ ZERO });
      } else if (comp == em::bx3) {
        return init_flds.bx3({ ZERO });
      } else {
        raise::Error("Invalid component", HERE);
        return ZERO;
      }
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

    void CustomPostStep(std::size_t step, long double time, Domain<S, M>& domain) {

      // check if the injector should be active
      if (step % injection_frequency != 0) {
        return;
      }

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
                          drift_ux / math::sqrt(1 + SQR(drift_ux)) * dt -
                          injection_frequency * dt,
                          x_init + injector_velocity * (dt_inj + dt) });
        } else {
          box.push_back(Range::All);
        }
      }

      // define indice range to reset fields
      boundaries_t<bool> incl_ghosts;
      for (auto d = 0; d < M::Dim; ++d) {
        incl_ghosts.push_back({ true, true });
      }
      auto fields_box = box;
      fields_box[0].second += injection_frequency * dt;
      const auto extent = domain.mesh.ExtentToRange(fields_box, incl_ghosts);
      tuple_t<std::size_t, M::Dim> x_min { 0 }, x_max { 0 };
      for (auto d = 0; d < M::Dim; ++d) {
        x_min[d] = extent[d].first;
        x_max[d] = extent[d].second;
      }

      // reset fields
      std::vector<unsigned short> comps = { em::bx1, em::bx2, em::bx3, 
                                            em::ex1, em::ex2, em::ex3 };

      // loop over all components
      for (const auto& comp : comps) {

        // get initial field value of component
        auto value = ResetFields((em)comp);

        if constexpr (M::Dim == Dim::_1D) {
          Kokkos::deep_copy(Kokkos::subview(domain.fields.em,
                                            std::make_pair(x_min[0], x_max[0]),
                                            comp),
                            value);
        } else if constexpr (M::Dim == Dim::_2D) {
          Kokkos::deep_copy(Kokkos::subview(domain.fields.em,
                                            std::make_pair(x_min[0], x_max[0]),
                                            std::make_pair(x_min[1], x_max[1]),
                                            comp),
                            value);
        } else if constexpr (M::Dim == Dim::_3D) {
          Kokkos::deep_copy(Kokkos::subview(domain.fields.em,
                                            std::make_pair(x_min[0], x_max[0]),
                                            std::make_pair(x_min[1], x_max[1]),
                                            std::make_pair(x_min[2], x_max[2]),
                                            comp),
                            value);
        } else {
          raise::Error("Invalid dimension", HERE);
        }
      }

      /* 
        tag particles inside the injection zone as dead 
      */

      // loop over particle species
      for (std::size_t s { 0 }; s < 2; ++s) {

        // get particle properties
        auto& species = domain.species[s];
        auto i1 = species.i1;
        auto tag = species.tag;

        // tag all particles with x > box[0].first as dead
        Kokkos::parallel_for(
          "RemoveParticles",
          species.rangeActiveParticles(),
          Lambda(index_t p) {
            // check if the particle is already dead
            if (tag(p) == ParticleTag::dead) {
              return;
            }
            // select the x-coordinate index
            auto x_i1 = i1(p);
            // check if the particle is inside the box of new plasma
            if (x_i1 > x_min[0]) {
              tag(p) = ParticleTag::dead;
            }
          }
        );
      }

      /* 
          Inject piston of fresh plasma
      */

      // same maxwell distribution as above
      const auto energy_dist  = arch::Maxwellian<S, M>(domain.mesh.metric,
                                                      domain.random_pool,
                                                      temperature,
                                                      -drift_ux,
                                                      in::x1);
      // spatial distribution of the particles
      // -> hack to use the uniform distribution in NonUniformInjector
      const auto spatial_dist = arch::Piston<S, M>(domain.mesh.metric,
                                                   box[0].first,
                                                   box[0].second,
                                                   in::x1);

      // inject piston of fresh plasma
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
