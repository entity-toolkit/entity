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

#include "global.h"

#include "traits/metric.h"

#include "framework/containers/particles.h"

/* -------------------------------------------------------------------------- */
/* Local macros    (same as in particle_pusher_sr.hpp)                        */
/* -------------------------------------------------------------------------- */
#define from_Xi_to_i(XI, I)                                                    \
  {                                                                            \
    (I) = static_cast<int>(((XI) + 1)) - 1;                                    \
  }

#define from_Xi_to_i_di(XI, I, DI)                                             \
  {                                                                            \
    from_Xi_to_i((XI), (I));                                                   \
    (DI) = static_cast<prtldx_t>((XI)) - static_cast<prtldx_t>(I);             \
  }

#define i_di_to_Xi(I, DI) (static_cast<real_t>((I)) + static_cast<real_t>((DI)))

/* -------------------------------------------------------------------------- */

namespace arch {

  /**
   * @brief Checks whether a particle reflects off a moving piston, called after particle has been moved by regular pusher
   *
   * @param p Index of particle
   * @param dt Timestep
   * @param particles Particle data arrays
   * @param metric Metric object for coordinate transformations
   * @param piston_position Position of the piston at the start of timestep in global coordinates
   * @param piston_v Velocity of piston at current timestep
   * @param is_left Is piston on the left side of the box or right side of the box
   */
  template <CartesianMetricClass M>
  Inline bool CrossesPiston(index_t                    p,
                            real_t                     dt,
                            const ntt::ParticleArrays& particles,
                            const M&                   metric,
                            real_t                     piston_position,
                            real_t                     piston_v,
                            bool                       is_left) {
    const real_t x1_Cd = i_di_to_Xi(particles.i1(p), particles.dx1(p));
    // x1_Cd_wallmove is not the actual particle coordinate
    // it is particle position minus how much the wall has moved in this
    // timestep This is a computational trick
    const real_t x1_Cd_wallmove =
      x1_Cd - metric.template transform<1, Idx::XYZ, Idx::U>({}, piston_v) * dt;
    const real_t x1_Ph_wallmove = metric.template convert<1, Crd::Cd, Crd::Ph>(
      x1_Cd_wallmove);

    return is_left ? piston_position > x1_Ph_wallmove
                   : piston_position < x1_Ph_wallmove;
  }

  /**
   * @brief Updates particle position and velocity if it reflects off a moving
   * piston, called to correct patticle position
   *
   * @param p Index of particle
   * @param dt Timestep
   * @param particles Particle data arrays
   * @param metric Metric object for coordinate transformations
   * @param piston_position Position of the piston at the start of timestep in global coordinates
   * @param piston_v Velocity of piston at current timestep
   * @param massive Whether the particle is massive or massless (e.g. photon)
   */
  template <CartesianMetricClass M>
  Inline void Piston(index_t                    p,
                     real_t                     dt,
                     const ntt::ParticleArrays& particles,
                     const M&                   metric,
                     real_t                     piston_position,
                     real_t                     piston_v,
                     bool                       massive) {

    // check if particle actually crosses the piston, if not return
    if (!CrossesPiston<M>(p, dt, particles, metric, piston_position, piston_v, true)) {
      return;
    }
    // step 1: calculate the particle 3 velocity
    const real_t gamma_p {
      massive ? U2GAMMA(particles.ux1(p), particles.ux2(p), particles.ux3(p))
              : NORM(particles.ux1(p), particles.ux2(p), particles.ux3(p))
    };

    const real_t beta_x_p = particles.ux1(p) / gamma_p;
    const real_t xp_prev = i_di_to_Xi(particles.i1_prev(p), particles.dx1_prev(p));

    const real_t piston_position_local = metric.template convert<1, Crd::Ph, Crd::Cd>(
      piston_position);
    int      i_w;
    prtldx_t dx_w;

    from_Xi_to_i_di(piston_position_local, i_w, dx_w);

    const real_t piston_gamma = ONE / math::sqrt(ONE - SQR(piston_v));

    // step 2: calculate the time for the particle to reach the piston
    const int      delta_i1_to_piston  = (i_w - particles.i1_prev(p));
    const prtldx_t delta_dx1_to_piston = (dx_w - particles.dx1_prev(p));
    const real_t dx_to_piston = i_di_to_Xi(delta_i1_to_piston, delta_dx1_to_piston);

    const real_t dt_to_piston = dx_to_piston /
                                metric.template transform<1, Idx::XYZ, Idx::U>(
                                  {},
                                  beta_x_p - piston_v);
    // step 3: calculate remaining time after the collision
    const real_t remaining_dt = dt - dt_to_piston;

    // step 4: update the particle's velocity and position after the collision
    particles.ux1(p) = -SQR(piston_gamma) *
                       ((ONE + SQR(piston_v)) * particles.ux1(p) -
                        TWO * piston_v * gamma_p);

    const real_t remaining_dt_inv_energy {
      massive ? (remaining_dt /
                 U2GAMMA(particles.ux1(p), particles.ux2(p), particles.ux3(p)))
              : (remaining_dt /
                 NORM(particles.ux1(p), particles.ux2(p), particles.ux3(p)))
    };
    // define piston integer and fractional coordinate
    int      i_w_coll  = i_w;
    prtldx_t dx_w_coll = dx_w + metric.template transform<1, Idx::XYZ, Idx::U>(
                                  {},
                                  piston_v) *
                                  dt_to_piston;

    i_w_coll += static_cast<int>(dx_w_coll >= ONE) -
                static_cast<int>(dx_w_coll < ZERO);
    dx_w_coll -= static_cast<prtldx_t>(dx_w_coll >= ONE);
    dx_w_coll += static_cast<prtldx_t>(dx_w_coll < ZERO);

    particles.i1(p) = i_w_coll;
    particles.dx1(
      p) = metric.template transform<1, Idx::XYZ, Idx::U>({}, particles.ux1(p)) *
             remaining_dt_inv_energy +
           dx_w_coll;

    particles.i1(p) += static_cast<int>(particles.dx1(p) >= ONE) -
                       static_cast<int>(particles.dx1(p) < ZERO);
    particles.dx1(p) -= static_cast<prtldx_t>(particles.dx1(p) >= ONE);
    particles.dx1(p) += static_cast<prtldx_t>(particles.dx1(p) < ZERO);
  }

} // namespace arch

#undef from_Xi_to_i_di
#undef from_Xi_to_i
#undef i_di_to_Xi

#endif // ARCHETYPES_UTILS_H
