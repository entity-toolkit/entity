#include "global.h"
#include "simulation.h"

#include <plog/Log.h>

namespace ntt {

template<template<typename T> class D>
void Simulation<D>::mainloop() {
  PLOGD << "Simulation mainloop started.";
  for (real_t time {0}; time < m_sim_params.m_runtime; time += m_sim_params.m_timestep) {
    PLOGD << "t = " << time;
    faradayHalfsubstep(time);

    // BC b-fields

    // particlePushSubstep(time);

    depositSubstep(time);

    // BC particles
    // BC currents
    
    faradayHalfsubstep(time);
    ampereSubstep(time);
    addCurrentsSubstep(time);
  }
  PLOGD << "Simulation mainloop finished.";
}

template class ntt::Simulation<ntt::One_D>;
template class ntt::Simulation<ntt::Two_D>;
template class ntt::Simulation<ntt::Three_D>;

}
