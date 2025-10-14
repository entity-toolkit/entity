#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/field_setter.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

#include <algorithm>
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

    Metadomain<S, M>& global_domain;

    // domain properties
    const real_t    global_xmin, global_xmax;
    // gas properties
    const real_t    drift_ux, temperature, temperature_ratio, filling_fraction;
    // injector properties
    const real_t    injector_velocity;
    const simtime_t injection_start;
    const int       injection_frequency;
    // magnetic field properties
    real_t          Btheta, Bphi, Bmag;
    InitFields<D>   init_flds;

    inline PGen(const SimulationParams& p, Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain }
      , global_xmin { global_domain.mesh().extent(in::x1).first }
      , global_xmax { global_domain.mesh().extent(in::x1).second }
      , drift_ux { p.template get<real_t>("setup.drift_ux") }
      , temperature { p.template get<real_t>("setup.temperature") }
      , temperature_ratio { p.template get<real_t>("setup.temperature_ratio") }
      , Bmag { p.template get<real_t>("setup.Bmag", ZERO) }
      , Btheta { p.template get<real_t>("setup.Btheta", ZERO) }
      , Bphi { p.template get<real_t>("setup.Bphi", ZERO) }
      , init_flds { Bmag, Btheta, Bphi, drift_ux }
      , filling_fraction { p.template get<real_t>("setup.filling_fraction", 1.0) }
      , injector_velocity { p.template get<real_t>("setup.injector_velocity", 1.0) }
      , injection_start { static_cast<simtime_t>(
          p.template get<real_t>("setup.injection_start", 0.0)) }
      , injection_frequency { p.template get<int>("setup.injection_frequency",
                                                  100) } {}

    inline PGen() {}

    auto MatchFields(real_t) const -> InitFields<D> {
      return init_flds;
    }

    auto FixFieldsConst(const bc_in&,
                        const em& comp) const -> std::pair<real_t, bool> {
      if (comp == em::ex1) {
        return { init_flds.ex1({ ZERO }), true };
      } else if (comp == em::ex2) {
        return { ZERO, true };
      } else if (comp == em::ex3) {
        return { ZERO, true };
      } else if (comp == em::bx1) {
        return { init_flds.bx1({ ZERO }), true };
      } else if (comp == em::bx2) {
        return { init_flds.bx2({ ZERO }), true };
      } else if (comp == em::bx3) {
        return { init_flds.bx3({ ZERO }), true };
      } else {
        raise::Error("Invalid component", HERE);
        return { ZERO, false };
      }
    }

    inline void InitPrtls(Domain<S, M>& domain) {

      /*
       *  Plasma setup as partially filled box
       *
       *  Plasma setup:
       *
       * global_xmin                            global_xmax
       * |                                      |
       * V                                      V
       * |:::::::::::|..........................|
       *             ^
       *             |
       *        filling_fraction
       */

      // minimum and maximum position of particles
      // real_t xg_min = global_xmin;
      // real_t xg_max = global_xmin + filling_fraction * (global_xmax - global_xmin);
      //
      // // define box to inject into
      // boundaries_t<real_t> box;
      // // loop over all dimensions
      // for (auto d { 0u }; d < (unsigned int)M::Dim; ++d) {
      //   // compute the range for the x-direction
      //   if (d == static_cast<decltype(d)>(in::x1)) {
      //     box.push_back({ xg_min, xg_max });
      //   } else {
      //     // inject into full range in other directions
      //     box.push_back(Range::All);
      //   }
      // }

      // species #1 -> e^-
      // species #2 -> protons

      // energy distribution of the particles
      // const auto temperatures = std::make_pair(temperature,
      //                                          temperature_ratio * temperature);
      // const auto drifts       = std::make_pair(
      //   std::vector<real_t> { -drift_ux, ZERO, ZERO },
      //   std::vector<real_t> { -drift_ux, ZERO, ZERO });
      // arch::InjectUniformMaxwellians<S, M>(params,
      //                                      domain,
      //                                      ONE,
      //                                      temperatures,
      //                                      { 1, 2 },
      //                                      drifts,
      //                                      false,
      //                                      box);
      // const auto maxwellian_1 = arch::experimental::Maxwellian<S, M>(
      //   domain.mesh.metric,
      //   domain.random_pool,
      //   temperatures.first,
      //   { -drift_ux, ZERO, ZERO });
      // const auto maxwellian_2 = arch::experimental::Maxwellian<S, M>(
      //   domain.mesh.metric,
      //   domain.random_pool,
      //   temperatures.second,
      //   { -drift_ux, ZERO, ZERO });
      // arch::InjectReplenishConst<S, M, decltype(maxwellian_1), decltype(maxwellian_2)>(
      //   params,
      //   domain,
      //   ONE,
      //   { 1, 2 },
      //   maxwellian_1,
      //   maxwellian_2,
      //   0.95,
      //   box);
    }

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {

      /*
       *  Replenish plasma in a moving injector
       *
       *  Injector setup:
       *
       * global_xmin           purge/replenish  global_xmax
       * |         x_init            |          |
       * V           v               V          V
       * |:::::::::::;::::::::::|\\\\\\\\|......|
       *                       xmin    xmax
       *                                 ^
       *                                 |
       *                           moving injector
       */

      // check if the injector should be active
      if ((step % injection_frequency) != 0 or
          ((time < injection_start) and (step != 0))) {
        return;
      }

      const auto dt = params.template get<real_t>("algorithms.timestep.dt");

      // initial position of injector
      const auto x_init = global_xmin +
                          filling_fraction * (global_xmax - global_xmin);

      // compute the position of the injector after the current timestep
      auto xmax = x_init + injector_velocity * (time + dt - injection_start);
      if (xmax >= global_xmax) {
        xmax = global_xmax;
      }

      // compute the beginning of the injected region
      auto xmin = xmax - 2.2 * static_cast<real_t>(injection_frequency) * dt;
      if (xmin <= global_xmin) {
        xmin = global_xmin;
      }
      if (step == 0) {
        xmin = global_xmin;
      }

      /*
          Inject slab of fresh plasma
      */

      // define box to inject into
      boundaries_t<real_t> inj_box;
      // loop over all dimension
      for (auto d = 0u; d < M::Dim; ++d) {
        if (d == 0) {
          inj_box.push_back({ xmin, xmax });
        } else {
          inj_box.push_back(Range::All);
        }
      }

      // same maxwell distribution as above
      const auto mass_1        = domain.species[0].mass();
      const auto mass_2        = domain.species[1].mass();
      const auto temperature_1 = temperature / mass_1;
      const auto temperature_2 = temperature * temperature_ratio / mass_2;

      const auto maxwellian_1 = arch::experimental::Maxwellian<S, M>(
        domain.mesh.metric,
        domain.random_pool,
        temperature_1,
        { -drift_ux, ZERO, ZERO });
      const auto maxwellian_2 = arch::experimental::Maxwellian<S, M>(
        domain.mesh.metric,
        domain.random_pool,
        temperature_2,
        { -drift_ux, ZERO, ZERO });

      const auto& mesh    = domain.mesh;
      const auto  inj_vel = injector_velocity;

      for (auto& s : { 1u, 2u }) {
        auto& species = domain.species[s - 1];
        auto  i1      = species.i1;
        auto  dx1     = species.dx1;
        auto  ux1     = species.ux1;
        auto  ux2     = species.ux2;
        auto  ux3     = species.ux3;
        auto  tag     = species.tag;

        Kokkos::parallel_for(
          "ResetParticles",
          species.rangeActiveParticles(),
          Lambda(index_t p) {
            if (tag(p) == ParticleTag::dead) {
              return;
            }
            const auto x_Cd = static_cast<real_t>(i1(p)) +
                              static_cast<real_t>(dx1(p));
            const auto x_Ph = mesh.metric.template convert<1, Crd::Cd, Crd::XYZ>(
              x_Cd);

            const coord_t<M::Dim> x_dummy { ZERO };
            if (x_Ph > xmin and x_Ph <= xmax) {
              vec_t<Dim::_3D> v_T { ZERO }, v_Cd { ZERO };
              if (s == 1u) {
                maxwellian_1(x_dummy, v_T, s);
              } else {
                maxwellian_2(x_dummy, v_T, s);
              }
              mesh.metric.template transform_xyz<Idx::T, Idx::XYZ>(x_dummy, v_T, v_Cd);
              ux1(p) = v_Cd[0];
              ux2(p) = v_Cd[1];
              ux3(p) = v_Cd[2];
            } else if (x_Ph > xmax) {
              ux1(p) = TWO * inj_vel /
                       math::max(static_cast<real_t>(1e-2),
                                 math::sqrt(ONE - SQR(inj_vel)));
              ux2(p) = ZERO;
              ux3(p) = ZERO;
            }
          });
      }

      arch::InjectReplenishConst<S, M, decltype(maxwellian_1), decltype(maxwellian_2)>(
        params,
        domain,
        ONE,
        { 1, 2 },
        maxwellian_1,
        maxwellian_2,
        1.0,
        inj_box);
    }
  };

} // namespace user
#endif
