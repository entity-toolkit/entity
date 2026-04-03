/**
 * @file archetypes/piston.h
 * @brief Piston functions for implementing the piston in the CustomPrtlUpdate in pgran
 * @implements
 *   - arch::PistonUpdate<> -> void
 *   - arch::CrossesPiston<> -> Bool
 * @namespaces:
 *   - arch::
 */

#ifndef ARCHETYPES_PISTON_H
#define ARCHETYPES_PISTON_H

#include "enums.h"
#include "global.h"

#include "archetypes/energy_dist.h"
#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"
#include "framework/domain/metadomain.h"

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

namespace arch {

  /**
   * @brief Updates particle position and velocity if it reflects off a moiving piston, called to correct patticle position
   * 
   * @param p Index of particle
   * @param pusher Particle pusher engine for particle update
   * @param piston_position Position of the piston at the start of timestep in global coordinates
   * @param piston_v Velocity of piston at current timestep
   */
  template <SimEngine::type S, class M, class PusherKernel>
  Inline void PistonUpdate(const index_t p,
                    const PusherKernel& pusher,
                    const real_t piston_position,
                    const real_t piston_v, 
                    const bool massive){
    
    coord_t<M::PrtlDim> xp_Cd { ZERO };
    pusher.getParticlePosition(p, xp_Cd);
    
    // step 1: calculate the particle 3 velocity
    const real_t gamma_p {
        massive
        ? (math::sqrt(ONE + SQR(pusher.ux1(p)) + SQR(pusher.ux2(p)) + SQR(pusher.ux3(p))))
        : (math::sqrt(SQR(pusher.ux1(p)) + SQR(pusher.ux2(p)) + SQR(pusher.ux3(p))))
        };
    
    const real_t beta_x_p = pusher.ux1(p)/gamma_p;
    const real_t xp_prev = i_di_to_Xi(pusher.i1_prev(p), pusher.dx1_prev(p));
    
    const real_t piston_position_local = pusher.metric.template convert<1, Crd::Ph, Crd::Cd>(piston_position);
    int i_w ; 
    prtldx_t  dx_w;

    from_Xi_to_i_di(piston_position_local, i_w, dx_w);

    const real_t piston_gamma = ONE/math::sqrt(ONE-SQR(piston_v));

    // step 2: calculate the time for the particle to reach the piston
    const int delta_i1_to_piston = (i_w - pusher.i1_prev(p));
    const prtldx_t delta_dx1_to_piston = (dx_w-pusher.dx1_prev(p));
    const real_t dx_to_piston = i_di_to_Xi(delta_i1_to_piston, delta_dx1_to_piston);
    
    const real_t dt_to_piston = dx_to_piston / pusher.metric.template transform<1, Idx::XYZ, Idx::U>(xp_Cd, beta_x_p-piston_v);
    // step 3: calculate remaining time after the collision
    const real_t remaining_dt = pusher.dt - dt_to_piston;
    // step 4: update the particle's velocity and position after the collision

    pusher.ux1(p) = -SQR(piston_gamma)*( (ONE+SQR(piston_v))*pusher.ux1(p)-TWO * piston_v*gamma_p );

        
    const real_t remaining_dt_inv_energy {
        massive
        ? (remaining_dt / math::sqrt(ONE + SQR(pusher.ux1(p)) + SQR(pusher.ux2(p)) + SQR(pusher.ux3(p))))
        : (remaining_dt / math::sqrt(SQR(pusher.ux1(p)) + SQR(pusher.ux2(p)) + SQR(pusher.ux3(p))))
        };
    // define piston integer and fractional coordinate

    int i_w_coll = i_w;
    prtldx_t dx_w_coll = dx_w + pusher.metric.template transform<1, Idx::XYZ, Idx::U>(xp_Cd, piston_v)*dt_to_piston;

    i_w_coll += static_cast<int>( dx_w_coll >= ONE) -
                   static_cast<int>( dx_w_coll< ZERO);
    dx_w_coll -= (dx_w_coll >= ONE);
    dx_w_coll+= (dx_w_coll< ZERO);

    pusher.i1(p) = i_w_coll;
    pusher.dx1(p) = pusher.metric.template transform<1, Idx::XYZ, Idx::U>(xp_Cd, pusher.ux1(p))*remaining_dt_inv_energy+dx_w_coll ;

    pusher.i1(p) += static_cast<int>(pusher.dx1(p) >= ONE) -
                   static_cast<int>(pusher.dx1(p) < ZERO);
    pusher.dx1(p) -= (pusher.dx1(p) >= ONE);
    pusher.dx1(p) += (pusher.dx1(p) < ZERO);


    }
  /**
   * @brief Checks whether a particle reflects off a moving piston, called after particle has been moved by regular pusher
   * 
   * @param p Index of particle
   * @param pusher Particle pusher engine for particle update
   * @param piston_position Position of the piston at the start of timestep in global coordinates
   * @param piston_v Velocity of piston at current timestep
   * @param is_left Is piston on the left side of the box or right side of the box
   */
  template <SimEngine::type S, class M, class PusherKernel>
  Inline bool CrossesPiston(const index_t p,
                    const PusherKernel& pusher,
                    const real_t piston_position,
                    const real_t piston_v,
                    const bool is_left ){
    const real_t x1_Cd = i_di_to_Xi(pusher.i1(p), pusher.dx1(p));
    coord_t<M::PrtlDim> xp_Cd { ZERO };
    pusher.getParticlePosition(p, xp_Cd);
    // x1_Cd_wallmove is not the actual particle coordinate
    // it is particle position minus how much the wall has moved in this timestep
    // This is a computational trick
    const real_t x1_Cd_wallmove = x1_Cd-pusher.metric.template transform<1, Idx::XYZ, Idx::U>(xp_Cd, piston_v)*pusher.dt;
    const real_t x1_Ph_wallmove = pusher.metric.template convert<1, Crd::Cd, Crd::Ph>(x1_Cd_wallmove);

    
    if (is_left){ // if piston is moving from left, ask if particle is to the left of piston
        if(piston_position>x1_Ph_wallmove){
            return true;
        } else {
            return false;
        }
    } else { // if piston is moving from the right, so ask is particle to right of piston
        if(piston_position<x1_Ph_wallmove){
            return true;
        } else {
            return false;
        }
    }


        

  }


} // namespace arch

 

#undef from_Xi_to_i_di
#undef from_Xi_to_i
#undef i_di_to_Xi

#endif // ARCHETYPES_UTILS_H
