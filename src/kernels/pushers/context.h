#ifndef KERNELS_PUSHERS_CONTEXT_H
#define KERNELS_PUSHERS_CONTEXT_H

#include "enums.h"
#include "global.h"

#include "utils/param_container.h"

namespace kernel {
  using namespace ntt;

  struct PusherContext {
    // species index
    spidx_t             species_index;
    // pusher algorithm(s) assigned to the species
    ParticlePusherFlags pusher_flags { ParticlePusher::NONE };
    // radiative drag force(s) enabled for the species
    RadiativeDragFlags  radiative_drag_flags { RadiativeDrag::NONE };

    // species parameters
    float mass, charge;

    // time variable
    simtime_t time;

    // global constants
    real_t dt, omegaB0;

    // grid parameters
    int                  ni1, ni2, ni3;
    boundaries_t<PrtlBC> boundaries;

    // parameters for the advanced features
    prm::Parameters gca_params;
    prm::Parameters radiative_drag_params;
    prm::Parameters atmosphere_params;
  };

  struct PusherArrays {
    spidx_t            sp;
    array_t<int*>      i1, i2, i3;
    array_t<int*>      i1_prev, i2_prev, i3_prev;
    array_t<prtldx_t*> dx1, dx2, dx3;
    array_t<prtldx_t*> dx1_prev, dx2_prev, dx3_prev;
    array_t<real_t*>   ux1, ux2, ux3;
    array_t<real_t*>   phi;
    array_t<real_t*>   weight;
    array_t<short*>    tag;
  };

} // namespace kernel

#endif // KERNELS_PUSHERS_CONTEXT_H