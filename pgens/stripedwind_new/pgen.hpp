#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/field_setter.h"
#include "archetypes/problem_generator.h"
#include "archetypes/utils.h"
#include "archetypes/spatial_dist.h"
#include "framework/domain/metadomain.h"

#include <algorithm>
#include <utility>

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct SWCurrentSheet : public arch::SpatialDistribution<S, M> {
    SWCurrentSheet(const M& metric,
                   real_t   alpha_s,
                   real_t   lambda_s,
                   real_t   delta_s,
                   real_t   drift_beta_x,
                   real_t   time)
      : arch::SpatialDistribution<S, M> { metric }
      , alpha_s { alpha_s }
      , lambda_s { lambda_s }
      , delta_s { delta_s }
      , drift_beta_x { drift_beta_x }
      , time { time } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      return ONE / SQR(math::cosh(
                     lambda_s / (constant::TWO_PI * delta_s) *
                     (alpha_s +
                      math::cos(constant::TWO_PI * (x_Ph[0] + drift_beta_x * time) / lambda_s))));
    }

  private:
    real_t alpha_s, lambda_s, delta_s, drift_beta_x, time;
  };

  template <Dimension D>
  struct InitBfields {
    InitBfields(real_t bmag,
                real_t btheta,
                real_t bphi,
                real_t drift_beta_x,
                real_t lambda_s,
                real_t delta_s,
                real_t alpha_s,
                real_t time = ZERO)
      : Bmag { bmag }
      , Btheta { btheta * static_cast<real_t>(convert::deg2rad) }
      , Bphi { bphi * static_cast<real_t>(convert::deg2rad) }
      , drift_beta_x { drift_beta_x }
      , Lambda_s { lambda_s }
      , Delta_s { delta_s }
      , Alpha_s { alpha_s }
      , time { time } {}

    // magnetic field components

    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {

      return Bmag *
             math::tanh(Lambda_s / (constant::TWO_PI * Delta_s) *
                        (Alpha_s + math::cos(constant::TWO_PI *
                                             (x_Ph[0] + time * drift_beta_x) / Lambda_s)));
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    const real_t Btheta, Bphi, drift_beta_x, Bmag, Delta_s, Alpha_s, Lambda_s;
    const real_t time;
  };

  template <Dimension D>
  struct InitFields : public InitBfields<D> {
    /*
      Sets up magnetic and electric field components for the simulation.
      Must satisfy E = -v x B for Lorentz Force to be zero.

      @param bmag: magnetic field scaling
      @param btheta: magnetic field polar angle
      @param bphi: magnetic field azimuthal angle
      @param drift_ux: drift velocity in the x direction
    */
    InitFields(real_t bmag,
               real_t btheta,
               real_t bphi,
               real_t drift_beta_x,
               real_t lambda_s,
               real_t delta_s,
               real_t alpha_s,
               real_t time = ZERO)
      : InitBfields<D> { bmag,     btheta,  bphi,    drift_beta_x,
                      lambda_s, delta_s, alpha_s, time } {}

    // magnetic field components

    // electric field components
    Inline auto ex1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t {
      return -this->drift_beta_x * this->Bmag *
             math::tanh(
               this->Lambda_s / (constant::TWO_PI * this->Delta_s) *
               (this->Alpha_s + math::cos(constant::TWO_PI * (x_Ph[0]) / this->Lambda_s)));
    }
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
    const real_t  global_xmin, global_xmax;
    // gas properties
    const real_t  drift_ux,  drift_gamma, drift_beta_x,  temperature, temperature_ratio, filling_fraction;
    // injector properties
    const real_t  injector_velocity, injection_start, dt;
    const int     injector_interval, delta_x_resetb;
    // magnetic field properties
    real_t        Btheta, Bphi, Bmag, Alpha_s, Lambda_s, Delta_s, eta_s;
    InitFields<D> init_flds;

    real_t injector_x;

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
      , Delta_s { p.template get<real_t>("setup.delta_s", 5.0) }
      , Lambda_s { p.template get<real_t>("setup.lambda_s", 100.0) }
      , Alpha_s { p.template get<real_t>("setup.alpha_s", ZERO) }
      , eta_s {p.template get<real_t>("setup.eta_s", 3.0) } // this is the ratio of the density in current sheet to cold wind 
      , drift_gamma {static_cast<real_t>(math::sqrt(1.+SQR(drift_ux)))}
      , drift_beta_x {drift_ux/drift_gamma}
      , init_flds { Bmag, Btheta, Bphi, drift_beta_x, Lambda_s, Delta_s, Alpha_s }
      , filling_fraction { p.template get<real_t>("setup.filling_fraction", 1.0) }
      , injector_velocity { p.template get<real_t>("setup.injector_velocity", 1.0) }
      , injection_start { p.template get<real_t>("setup.injection_start", 0.0) }
      , injector_interval { p.template get<int>("setup.injector_interval", 100) }
      , delta_x_resetb {p.template get<int>("setup.delta_x_resetb", 50) }
      , dt { p.template get<real_t>("algorithms.timestep.dt") } {}
      

    inline PGen() {}

    //real_t drift_gamma = math::sqrt(1.+SQR(drift_ux));

    //real_t drift_beta_x = drift_ux / drift_gamma;

    auto MatchFields(real_t time) const -> InitBfields<D> {
      const auto init_bflds =
        InitBfields<D>(Bmag, Btheta, Bphi, drift_beta_x, Lambda_s, Delta_s, Alpha_s, time);
      return init_bflds;
    }

    // auto FixFieldsConst(const bc_in&,
    //                     const em& comp) const -> std::pair<real_t, bool> {
    //   if (comp == em::ex1) {
    //     return { init_flds.ex1({ ZERO }), true };
    //   } else if (comp == em::ex2) {
    //     return { ZERO, true };
    //   } else if (comp == em::ex3) {
    //     return { ZERO, true };
    //   } else if (comp == em::bx1) {
    //     return { init_flds.bx1({ ZERO }), true };
    //   } else if (comp == em::bx2) {
    //     return { init_flds.bx2({ ZERO }), true };
    //   } else if (comp == em::bx3) {
    //     return { init_flds.bx3({ ZERO }), true };
    //   } else {
    //     raise::Error("Invalid component", HERE);
    //     return { ZERO, false };
    //   }
    // }

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

      const auto sigma = params.template get<real_t>("scales.sigma0");
      const auto c_omp = params.template get<real_t>("scales.skindepth0");

      const auto cs_temperature = HALF * sigma / eta_s;

      real_t xg_min = global_xmin;
      real_t xg_max = global_xmin + filling_fraction * (global_xmax - global_xmin);

      // define box to inject into
      boundaries_t<real_t> box;
      // loop over all dimensions
      for (auto d { 0u }; d < (unsigned int)M::Dim; ++d) {
        // compute the range for the x-direction
        if (d == static_cast<decltype(d)>(in::x1)) {
          box.push_back({ xg_min, xg_max });
        } else {
          // inject into full range in other directions
          box.push_back(Range::All);
        }
      }

      // define temperatures of species
      const auto temperatures = std::make_pair(temperature,
                                               temperature_ratio * temperature);
      // define drift speed of species
      const auto drifts       = std::make_pair(
        std::vector<real_t> { -drift_ux, ZERO, ZERO },
        std::vector<real_t> { -drift_ux, ZERO, ZERO });

      // inject particles
      arch::InjectUniformMaxwellians<S, M>(params,
                                           domain,
                                           ONE,
                                           temperatures,
                                           { 1, 2 },
                                           drifts,
                                           false,
                                           box);


      

      const auto  edist_cs = arch::Maxwellian<S, M>(domain.mesh.metric, domain.random_pool, cs_temperature, { -drift_ux, ZERO, ZERO } );

      const auto sdist_cs = SWCurrentSheet<S, M>(domain.mesh.metric, Alpha_s, Lambda_s, Delta_s, drift_beta_x, ZERO);

      // inject particles in current sheet, need to include classes of energy distribution and spatial distribution
      arch::InjectNonUniform<S, M, decltype(edist_cs), decltype(edist_cs), decltype(sdist_cs)>(
                                    params,
                                    domain,
                                    {1, 2},
                                    {edist_cs, edist_cs},
                                    sdist_cs, 
                                    eta_s,
                                    false, 
                                    box);
      injector_x = xg_max;
    }

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {

      if (step % injector_interval == 0) {
        const auto dt = params.template get<real_t>("algorithms.timestep.dt");
        const auto new_injector_x = injector_x +
                                    injector_velocity * injector_interval * dt;
        const auto  xmin = injector_x - injector_interval * dt;
        /**
         * tag particles inside the injection zone as dead
         **/
        const auto& mesh = domain.mesh;
        // loop over particle species
        for (auto s { 0u }; s < 2; ++s) {
          // get particle properties
          auto& species = domain.species[s];
          auto  i1      = species.i1;
          auto  dx1     = species.dx1;
          auto  tag     = species.tag;

          Kokkos::parallel_for(
            "RemoveParticles",
            species.rangeActiveParticles(),
            Lambda(index_t p) {
              // check if the particle is already dead
              if (tag(p) == ParticleTag::dead) {
                return;
              }
              // convert particle position to grid coordinates
              const auto x_Cd = static_cast<real_t>(i1(p)) +
                                static_cast<real_t>(dx1(p));
              // convert grid coordinates to physical coordinates
              const auto x_Ph = mesh.metric.template convert<1, Crd::Cd, Crd::XYZ>(
                x_Cd);

              // if the particle position is to the right of xmin, tag it as dead
              if (x_Ph > xmin) {
                tag(p) = ParticleTag::dead;
              }
            });
        }

        /*
          Reset the fields inside the purged region
        */
        // define indices range to reset fields
        // (not including ghost zones in either direction)
        boundaries_t<bool> incl_ghosts;
        for (auto d = 0; d < M::Dim; ++d) {
          incl_ghosts.push_back({ false, false });
        }

        // define the rectangular box region where fields are reset
        boundaries_t<real_t> purge_box;
        // loop over all dimension
        for (auto d = 0u; d < M::Dim; ++d) {
          if (d == 0) {
            purge_box.push_back({ xmin, new_injector_x });
          } else {
            purge_box.push_back(Range::All);
          }
        }

        // convert physical extent to a range of cells
        const auto extent = domain.mesh.ExtentToRange(purge_box, incl_ghosts);

        // record the range min/max boundaries in each dimension
        tuple_t<std::size_t, M::Dim> x_min { 0 }, x_max { 0 };

        // define a tuple to set the volume over which the magnetic field is reset to the right of the injector
        tuple_t<std::size_t, M::Dim> x_min_bzone { 0 }, x_max_bzone { 0 };

        for (auto d = 0; d < M::Dim; ++d) {
          x_min[d] = extent[d].first  ;
          x_max[d] = extent[d].second ;
          if (d == 0){
            x_min_bzone[d] = x_max[d]+1  ;
            x_max_bzone[d] = x_max[d]+delta_x_resetb ;
          } else {
            x_min_bzone[d] = x_min[d];
            x_max_bzone[d] = x_max[d];
          }
          
        }
        // I am re-setting the fields in the region where the plasma is being injected
        const auto init_bflds = InitBfields<D>(Bmag, Btheta, Bphi, drift_beta_x, Lambda_s, Delta_s, Alpha_s, time);

        Kokkos::parallel_for("ResetFields",
                             CreateRangePolicy<M::Dim>(x_min_bzone, x_max_bzone),
                             arch::SetEMFields_kernel<decltype(init_bflds), S, M> {
                               domain.fields.em, //                ^
                               init_bflds, // <-- but proper injector fields
                               domain.mesh.metric });
        
        // ADD FRESH PLASMA INJECTOR (COPY FROM BELOW) 
        // This is code taken from below
        // inject particles now
        boundaries_t<real_t> inj_box;
        //loop over all dimensions
        for (auto d = 0u; d<M::Dim; ++d){
          if (d == 0){
            inj_box.push_back({ xmin, injector_x });
          }
          else{
            inj_box.push_back(Range::All);
          }
        }
        
        

        

        const auto temperatures = std::make_pair(temperature, temperature_ratio*temperature);

        const auto drifts = std::make_pair(
          std::vector<real_t> {-drift_ux, ZERO, ZERO},
          std::vector<real_t> {-drift_ux, ZERO, ZERO});

        arch::InjectUniformMaxwellians<S, M>(params,
                                              domain,
                                              ONE,
                                              temperatures,
                                              {1, 2},
                                              drifts,
                                              false,
                                              inj_box);

        // Now inject the current sheet
        const auto sigma = params.template get<real_t>("scales.sigma0");
        const auto c_omp = params.template get<real_t>("scales.skindepth0");

        const auto cs_temperature = HALF * sigma / eta_s;

        const auto  edist_cs = arch::Maxwellian<S, M>(domain.mesh.metric, domain.random_pool, cs_temperature, { -drift_ux, ZERO, ZERO } );

        const auto sdist_cs = SWCurrentSheet<S, M>(domain.mesh.metric, Alpha_s, Lambda_s, Delta_s, drift_beta_x, ZERO);

        // inject particles in current sheet, need to include classes of energy distribution and spatial distribution
        arch::InjectNonUniform<S, M, decltype(edist_cs), decltype(edist_cs), decltype(sdist_cs)>(
                                    params,
                                    domain,
                                    {1, 2},
                                    {edist_cs, edist_cs},
                                    sdist_cs, 
                                    eta_s,
                                    false, 
                                    inj_box);

        injector_x = new_injector_x;
        //
      }
    }

    //
    // /*
    //  *  Replenish plasma in a moving injector
    //  *
    //  *  Injector setup:
    //  *
    //  * global_xmin           purge/replenish  global_xmax
    //  * |         x_init            |          |
    //  * V           v               V          V
    //  * |:::::::::::;::::::::::|\\\\\\\\|......|
    //  *                       xmin    xmax
    //  *                                 ^
    //  *                                 |
    //  *                           moving injector
    //  */
    //
    // // check if the injector should be active
    // if (step % injection_frequency != 0) {
    //   return;
    // }
    //
    // // initial position of injector
    // const auto x_init = global_xmin +
    //                     filling_fraction * (global_xmax - global_xmin);
    //
    // // compute the position of the injector after the current timestep
    // auto xmax = x_init + injector_velocity *
    //                        (std::max<real_t>(time - injection_start, ZERO) + dt);
    // if (xmax >= global_xmax) {
    //   xmax = global_xmax;
    // }
    //
    // // compute the beginning of the injected region
    // auto xmin = xmax - injection_frequency * dt;
    // if (xmin <= global_xmin) {
    //   xmin = global_xmin;
    // }
    //
    // // define indice range to reset fields
    // boundaries_t<bool> incl_ghosts;
    // for (auto d = 0; d < M::Dim; ++d) {
    //   incl_ghosts.push_back({ false, false });
    // }
    //
    // // define box to reset fields
    // boundaries_t<real_t> purge_box;
    // // loop over all dimension
    // for (auto d = 0u; d < M::Dim; ++d) {
    //   if (d == 0) {
    //     purge_box.push_back({ xmin, global_xmax });
    //   } else {
    //     purge_box.push_back(Range::All);
    //   }
    // }
    //
    // const auto extent = domain.mesh.ExtentToRange(purge_box, incl_ghosts);
    // tuple_t<std::size_t, M::Dim> x_min { 0 }, x_max { 0 };
    // for (auto d = 0; d < M::Dim; ++d) {
    //   x_min[d] = extent[d].first;
    //   x_max[d] = extent[d].second;
    // }
    //
    // Kokkos::parallel_for("ResetFields",
    //                      CreateRangePolicy<M::Dim>(x_min, x_max),
    //                      arch::SetEMFields_kernel<decltype(init_flds), S, M> {
    //                        domain.fields.em,
    //                        init_flds,
    //                        domain.mesh.metric });
    // global_domain.CommunicateFields(domain, Comm::E | Comm::B);
    //
    // /*
    //   tag particles inside the injection zone as dead
    // */
    // const auto& mesh = domain.mesh;
    //
    // // loop over particle species
    // for (auto s { 0u }; s < 2; ++s) {
    //   // get particle properties
    //   auto& species = domain.species[s];
    //   auto  i1      = species.i1;
    //   auto  dx1     = species.dx1;
    //   auto  tag     = species.tag;
    //
    //   Kokkos::parallel_for(
    //     "RemoveParticles",
    //     species.rangeActiveParticles(),
    //     Lambda(index_t p) {
    //       // check if the particle is already dead
    //       if (tag(p) == ParticleTag::dead) {
    //         return;
    //       }
    //       const auto x_Cd = static_cast<real_t>(i1(p)) +
    //                         static_cast<real_t>(dx1(p));
    //       const auto x_Ph = mesh.metric.template convert<1, Crd::Cd, Crd::XYZ>(
    //         x_Cd);
    //
    //       if (x_Ph > xmin) {
    //         tag(p) = ParticleTag::dead;
    //       }
    //     });
    // }
    //
    // /*
    //     Inject slab of fresh plasma
    // */
    //
    // // define box to inject into
    // boundaries_t<real_t> inj_box;
    // // loop over all dimension
    // for (auto d = 0u; d < M::Dim; ++d) {
    //   if (d == 0) {
    //     inj_box.push_back({ xmin, xmax });
    //   } else {
    //     inj_box.push_back(Range::All);
    //   }
    // }
    //
    // // same maxwell distribution as above
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
    //                                      inj_box);
  };

} // namespace user
#endif