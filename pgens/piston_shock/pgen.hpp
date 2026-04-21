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

/* -------------------------------------------------------------------------- */


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
    const real_t  cell_size;
    // gas properties
    const real_t  temperature, temperature_ratio;
    // injector properties
    const real_t  dt;
    // magnetic field properties
    real_t        Btheta, Bphi, Bmag;
    InitFields<D> init_flds;

    // piston properties
    const real_t piston_velocity;
    int i_piston, initial_i_piston;
    real_t di_piston, piston_velocity_cd, piston_position_cd, piston_position;

    
    inline PGen(const SimulationParams& p, Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain }
      , global_xmin { global_domain.mesh().extent(in::x1).first }
      , global_xmax { global_domain.mesh().extent(in::x1).second }
      , cell_size { (global_xmax - global_xmin) / global_domain.mesh().n_all(in::x1) }
      , temperature { p.template get<real_t>("setup.temperature") }
      , temperature_ratio { p.template get<real_t>("setup.temperature_ratio") }
      , piston_velocity {p.template get<real_t>("setup.piston_velocity", ZERO)}
      , Bmag { p.template get<real_t>("setup.Bmag", ZERO) }
      , Btheta { p.template get<real_t>("setup.Btheta", ZERO) }
      , Bphi { p.template get<real_t>("setup.Bphi", ZERO) }
      , dt { p.template get<real_t>("algorithms.timestep.dt") }
      , init_flds { Bmag, Btheta, Bphi, ZERO } {}

    inline PGen() {}

    auto MatchFields(simtime_t) const -> InitFields<D> {
      return init_flds;
    }

    inline void InitPrtls(Domain<S, M>& domain) {

      // set initial position of piston 
      piston_position = ZERO;
      piston_position_cd = domain.mesh.metric.template convert<1, Crd::Ph, Crd::Cd>(piston_position);
      // convert to cell units and get cell index and sub-cell position
      from_Xi_to_i_di(piston_position_cd, i_piston, di_piston);
      // store initial cell index of piston for window updates
      initial_i_piston = i_piston;

      // piston velocity in code units
      coord_t<M::PrtlDim> xp_Cd { ZERO };
      piston_velocity_cd = domain.mesh.metric.template transform<1, Idx::XYZ, Idx::U>(xp_Cd, piston_velocity);

      // define temperatures of species
      const auto temperatures = std::make_pair(temperature,
                                               temperature_ratio * temperature);

      // inject particles
      arch::InjectUniformMaxwellians<S, M>(params,
                                           domain,
                                           ONE,
                                           temperatures,
                                           { 1, 2 });
    }

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {

      /* 
        update piston position
      */
      // piston movement over timestep
      di_piston += piston_velocity_cd * dt;
      // check if the piston has moved to the next cell
      i_piston += static_cast<int>(di_piston >= ONE);
      // keep track of how much the piston has moved into the next cell
      di_piston -= (di_piston >= ONE);

      // check if the window should be moved
      if ((i_piston-initial_i_piston) >= N_GHOSTS) {
        
        // move the window and all fields and particles in it
        arch::MoveWindow<S, M, in::x1>(domain, global_domain, N_GHOSTS);
        
        /*
            Inject slab of fresh plasma
        */
        const real_t xmax = global_domain.mesh().extent(in::x1).second;
        const real_t xmin = xmax - N_GHOSTS * cell_size;
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

        // shift the piston indices back by the number of ghost cells to account for window movement
        i_piston -= N_GHOSTS;
      }

      // compute current position of piston
      piston_position_cd = i_di_to_Xi(i_piston, di_piston);
      // convert to physical coordinates
      piston_position = domain.mesh.metric.template convert<1, Crd::Cd, Crd::Ph>(piston_position_cd);
      
    }
    

    struct CustomPrtlUpdate {
      real_t x_piston;
      real_t v_piston;
      bool is_left;
      bool massive;
      
      template <class Coord, class PusherKernel>
      Inline void operator()(index_t p, Coord& xp, PusherKernel& pusher) const {

        if (arch::CrossesPiston<S, M, PusherKernel>(p, pusher, x_piston, v_piston, is_left)){
            arch::PistonUpdate<S, M, PusherKernel>(p, pusher, x_piston, v_piston, massive);
        }
      }
    };

    template <class D>
    auto CustomParticleUpdate(simtime_t time, spidx_t sp, D& domain) const {
      return CustomPrtlUpdate { piston_position, piston_velocity, true, true};
      };
  };

} // namespace user

#endif // PROBLEM_GENERATOR_H