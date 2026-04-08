#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/field_setter.h"
#include "archetypes/problem_generator.h"
#include "archetypes/traits.h"
#include "archetypes/utils.h"
#include "archetypes/piston.h"
#include "archetypes/moving_window.h"
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
    static constexpr auto engines {
      arch::traits::pgen::compatible_with<SimEngine::SRPIC>::value
    };
    static constexpr auto metrics {
      arch::traits::pgen::compatible_with<Metric::Minkowski>::value
    };
    static constexpr auto dimensions {
      arch::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
    };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    Metadomain<S, M>& global_domain;

    // domain properties
    const real_t  global_xmin, global_xmax;
    // gas properties
    const real_t  temperature, temperature_ratio;
    // injector properties
    const real_t  dt;
    // magnetic field properties
    real_t        Btheta, Bphi, Bmag;
    InitFields<D> init_flds;

    // piston properties
    const real_t piston_velocity;
    int i_piston;
    real_t di_piston, piston_position;

    // window properties
    const int window_update_frequency;

    inline PGen(const SimulationParams& p, Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain }
      , global_xmin { global_domain.mesh().extent(in::x1).first }
      , global_xmax { global_domain.mesh().extent(in::x1).second }
      , temperature { p.template get<real_t>("setup.temperature") }
      , temperature_ratio { p.template get<real_t>("setup.temperature_ratio") }
      , Bmag { p.template get<real_t>("setup.Bmag", ZERO) }
      , Btheta { p.template get<real_t>("setup.Btheta", ZERO) }
      , Bphi { p.template get<real_t>("setup.Bphi", ZERO) }
      , init_flds { Bmag, Btheta, Bphi, ZERO }
      , dt { p.template get<real_t>("algorithms.timestep.dt") }
      , window_update_frequency { p.template get<int>("setup.window_update_frequency", N_GHOSTS) }
      , piston_velocity { p.template get<real_t>("setup.piston_velocity", ZERO) 
            / math::sqrt(ONE - SQR(p.template get<real_t>("setup.piston_velocity", ZERO)))} {}

    inline PGen() {}

    auto MatchFields(simtime_t) const -> InitFields<D> {
      return init_flds;
    }

    inline void InitPrtls(Domain<S, M>& domain) {

      // set initial position of piston
      i_piston = 0;
      di_piston = ZERO;
      piston_position = ZERO;

      // define temperatures of species
      const auto temperatures = std::make_pair(temperature,
                                               temperature_ratio * temperature);
      // define drift speed of species
      const auto drifts = std::make_pair(std::vector<real_t> { ZERO, ZERO, ZERO },
                                         std::vector<real_t> { ZERO, ZERO, ZERO });

      // inject particles
      arch::InjectUniformMaxwellians<S, M>(params,
                                           domain,
                                           ONE,
                                           temperatures,
                                           { 1, 2 },
                                           drifts,
                                           false);
    }

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {

      /* 
        update piston position
      */
      // piston movement over timestep
      di_piston += piston_velocity * dt;
      // check if the piston has moved to the next cell
      i_piston += static_cast<int>(di_piston >= ONE);
      // keep track of how much the piston has moved into the next cell
      di_piston -= (di_piston >= ONE);

      // check if the window should be moved
      if (i_piston >= window_update_frequency) {

        // move the window and all fields and particles in it
        arch::MoveWindow<S, M, in::x1>(domain, window_update_frequency);

        // synch ghost zones after moving the window
        global_domain.CommunicateFields(domain, Comm::E | Comm::B);

        /*
            Inject slab of fresh plasma
        */
        const real_t cell_size = ZERO; // ToDo: get cell size from global domain
        const real_t xmax = global_xmax;
        const real_t xmin = xmax - window_update_frequency * cell_size;
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
        const auto temperatures = std::make_pair(temperature,
                                                temperature_ratio * temperature);
        const auto drifts       = std::make_pair(
          std::vector<real_t> { ZERO, ZERO, ZERO },
          std::vector<real_t> { ZERO, ZERO, ZERO });
        arch::InjectUniformMaxwellians<S, M>(params,
                                            domain,
                                            ONE,
                                            temperatures,
                                            { 1, 2 },
                                            drifts,
                                            false,
                                            inj_box);

        i_piston -= window_update_frequency;
      }

      // compute current position of piston
      piston_position = static_cast<real_t>(i_piston) + di_piston;
    }
    

    struct CustomPrtlUpdate {
      real_t x_piston;
      real_t v_piston;
      real_t xg_max;
      bool is_left;
      bool massive;
      
      template <class Coord, class PusherKernel>
      Inline void operator()(index_t p, Coord& xp, PusherKernel& pusher) const {

        real_t piston_position_use;

        if (x_piston < xg_max){ //make sure piston has not reached the right wall
            piston_position_use = x_piston;
        } else {
          piston_position_use = xg_max;
        }

        if (arch::CrossesPiston<S, M, PusherKernel>(p, pusher, piston_position_use, v_piston, is_left)){
            arch::PistonUpdate<S, M, PusherKernel>(p, pusher, piston_position_use, v_piston, massive);
        }
      }
    };

    template <class D>
    auto CustomParticleUpdate(simtime_t time, spidx_t sp, D& domain) const {
      return CustomPrtlUpdate { piston_position, piston_velocity, global_xmax, true, true};
      };
  };

} // namespace user

#endif // PROBLEM_GENERATOR_H