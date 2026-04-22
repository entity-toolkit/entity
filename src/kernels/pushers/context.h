#ifndef KERNELS_PUSHERS_CONTEXT_H
#define KERNELS_PUSHERS_CONTEXT_H

#include "enums.h"
#include "global.h"

#include "utils/numeric.h"

namespace kernel {
  using namespace ntt;

  struct PusherGCAContext {
    real_t larmor_max { ZERO };
    real_t e_ovr_b_sqr_max { ZERO };

    PusherGCAContext() = default;

    PusherGCAContext(real_t larmor_max, real_t e_ovr_b_max)
      : larmor_max { larmor_max }
      , e_ovr_b_sqr_max { SQR(e_ovr_b_max) } {}
  };

  struct PusherSynchrotronDragContext {
    real_t coeff { ZERO };

    PusherSynchrotronDragContext() = default;

    PusherSynchrotronDragContext(real_t dt, real_t omegaB0, real_t gamma_rad, float mass)
      : coeff { static_cast<real_t>(0.1) * dt * omegaB0 / (SQR(gamma_rad * mass)) } {
    }
  };

  struct PusherComptonDragContext {
    real_t coeff { ZERO };

    PusherComptonDragContext() = default;

    PusherComptonDragContext(real_t dt, real_t omegaB0, real_t gamma_rad, float mass)
      : coeff { static_cast<real_t>(0.1) * dt * omegaB0 / (SQR(gamma_rad * mass)) } {
    }
  };

  struct PusherAtmosphereContext {
    real_t gx1 { ZERO }, gx2 { ZERO }, gx3 { ZERO };
    real_t x_surf { ZERO }, ds { ZERO };

    PusherAtmosphereContext() = default;

    PusherAtmosphereContext(real_t gx1, real_t gx2, real_t gx3, real_t x_surf, real_t ds)
      : gx1 { gx1 }
      , gx2 { gx2 }
      , gx3 { gx3 }
      , x_surf { x_surf }
      , ds { ds } {}
  };

  struct PusherContext {
    // species index
    const spidx_t             species_index;
    // pusher algorithm(s) assigned to the species
    const ParticlePusherFlags pusher_flags { ParticlePusher::NONE };
    // radiative drag force(s) enabled for the species
    const RadiativeDragFlags  radiative_drag_flags { RadiativeDrag::NONE };

    // species parameters
    const float mass, charge;

    // time variable
    const simtime_t time;

    // global constants
    const real_t dt, omegaB0;

    // grid parameters
    const int ni1, ni2, ni3;

    PusherContext(const spidx_t             species_index,
                  const ParticlePusherFlags pusher_flags,
                  const RadiativeDragFlags  radiative_drag_flags,
                  float                     mass,
                  float                     charge,
                  simtime_t                 time,
                  real_t                    dt,
                  real_t                    omegaB0,
                  int                       ni1,
                  int                       ni2,
                  int                       ni3)
      : species_index { species_index }
      , pusher_flags { pusher_flags }
      , radiative_drag_flags { radiative_drag_flags }
      , mass { mass }
      , charge { charge }
      , time { time }
      , dt { dt }
      , omegaB0 { omegaB0 }
      , ni1 { ni1 }
      , ni2 { ni2 }
      , ni3 { ni3 } {}

    // parameters for the advanced features
    PusherGCAContext             gca;
    PusherSynchrotronDragContext synchrotron_drag;
    PusherComptonDragContext     compton_drag;
    PusherAtmosphereContext      atmosphere;
  };

  template <Dimension D>
  struct PusherBoundaries {
    bool is_absorb_i1min { false }, is_absorb_i1max { false };
    bool is_absorb_i2min { false }, is_absorb_i2max { false };
    bool is_absorb_i3min { false }, is_absorb_i3max { false };
    bool is_periodic_i1min { false }, is_periodic_i1max { false };
    bool is_periodic_i2min { false }, is_periodic_i2max { false };
    bool is_periodic_i3min { false }, is_periodic_i3max { false };
    bool is_reflect_i1min { false }, is_reflect_i1max { false };
    bool is_reflect_i2min { false }, is_reflect_i2max { false };
    bool is_reflect_i3min { false }, is_reflect_i3max { false };
    bool is_axis_i2min { false }, is_axis_i2max { false };

    PusherBoundaries(const boundaries_t<PrtlBC>& boundaries) {
      raise::ErrorIf(boundaries.size() < 1, "boundaries defined incorrectly", HERE);
      is_absorb_i1min = (boundaries[0].first == PrtlBC::ATMOSPHERE) ||
                        (boundaries[0].first == PrtlBC::ABSORB);
      is_absorb_i1max = (boundaries[0].second == PrtlBC::ATMOSPHERE) ||
                        (boundaries[0].second == PrtlBC::ABSORB);
      is_periodic_i1min = (boundaries[0].first == PrtlBC::PERIODIC);
      is_periodic_i1max = (boundaries[0].second == PrtlBC::PERIODIC);
      is_reflect_i1min  = (boundaries[0].first == PrtlBC::REFLECT);
      is_reflect_i1max  = (boundaries[0].second == PrtlBC::REFLECT);
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_absorb_i2min = (boundaries[1].first == PrtlBC::ATMOSPHERE) ||
                          (boundaries[1].first == PrtlBC::ABSORB);
        is_absorb_i2max = (boundaries[1].second == PrtlBC::ATMOSPHERE) ||
                          (boundaries[1].second == PrtlBC::ABSORB);
        is_periodic_i2min = (boundaries[1].first == PrtlBC::PERIODIC);
        is_periodic_i2max = (boundaries[1].second == PrtlBC::PERIODIC);
        is_reflect_i2min  = (boundaries[1].first == PrtlBC::REFLECT);
        is_reflect_i2max  = (boundaries[1].second == PrtlBC::REFLECT);
        is_axis_i2min     = (boundaries[1].first == PrtlBC::AXIS);
        is_axis_i2max     = (boundaries[1].second == PrtlBC::AXIS);
      }
      if constexpr (D == Dim::_3D) {
        raise::ErrorIf(boundaries.size() < 3, "boundaries defined incorrectly", HERE);
        is_absorb_i3min = (boundaries[2].first == PrtlBC::ATMOSPHERE) ||
                          (boundaries[2].first == PrtlBC::ABSORB);
        is_absorb_i3max = (boundaries[2].second == PrtlBC::ATMOSPHERE) ||
                          (boundaries[2].second == PrtlBC::ABSORB);
        is_periodic_i3min = (boundaries[2].first == PrtlBC::PERIODIC);
        is_periodic_i3max = (boundaries[2].second == PrtlBC::PERIODIC);
        is_reflect_i3min  = (boundaries[2].first == PrtlBC::REFLECT);
        is_reflect_i3max  = (boundaries[2].second == PrtlBC::REFLECT);
      }
    }
  };

  struct PusherArrays {
    array_t<int*>      i1, i2, i3;
    array_t<int*>      i1_prev, i2_prev, i3_prev;
    array_t<prtldx_t*> dx1, dx2, dx3;
    array_t<prtldx_t*> dx1_prev, dx2_prev, dx3_prev;
    array_t<real_t*>   ux1, ux2, ux3;
    array_t<real_t*>   phi;
    array_t<real_t*>   weight;
    array_t<short*>    tag;

    PusherArrays(spidx_t sp) : sp { sp } {}

    PusherArrays(spidx_t            sp,
                 array_t<int*>      i1,
                 array_t<int*>      i2,
                 array_t<int*>      i3,
                 array_t<int*>      i1_prev,
                 array_t<int*>      i2_prev,
                 array_t<int*>      i3_prev,
                 array_t<prtldx_t*> dx1,
                 array_t<prtldx_t*> dx2,
                 array_t<prtldx_t*> dx3,
                 array_t<prtldx_t*> dx1_prev,
                 array_t<prtldx_t*> dx2_prev,
                 array_t<prtldx_t*> dx3_prev,
                 array_t<real_t*>   ux1,
                 array_t<real_t*>   ux2,
                 array_t<real_t*>   ux3,
                 array_t<real_t*>   phi,
                 array_t<real_t*>   weight,
                 array_t<short*>    tag)
      : sp { sp }
      , i1 { i1 }
      , i2 { i2 }
      , i3 { i3 }
      , i1_prev { i1_prev }
      , i2_prev { i2_prev }
      , i3_prev { i3_prev }
      , dx1 { dx1 }
      , dx2 { dx2 }
      , dx3 { dx3 }
      , dx1_prev { dx1_prev }
      , dx2_prev { dx2_prev }
      , dx3_prev { dx3_prev }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , phi { phi }
      , weight { weight }
      , tag { tag } {}

  private:
    spidx_t sp;
  };

} // namespace kernel

#endif // KERNELS_PUSHERS_CONTEXT_H