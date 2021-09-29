#include "global.h"
#include "simulation.h"

#include <plog/Log.h>

namespace ntt {

template <template <typename T> class D> void Simulation<D>::mainloop() {
  PLOGD << "Simulation mainloop started.";
  unsigned long timax{static_cast<unsigned long>(m_sim_params.m_runtime / m_sim_params.m_timestep)};
  real_t time{0.0};
  for (unsigned long ti{0}; ti < timax; ++ti) {
    PLOGD << "t = " << time;
    faradayHalfsubstep(time);

    // BC b-fields

    // particlePushSubstep(time);

    // depositSubstep(time);

    // BC particles
    // BC currents

    faradayHalfsubstep(time);
    ampereSubstep(time);
    addCurrentsSubstep(time);
    resetCurrentsSubstep(time);

    time += m_sim_params.m_timestep;
  }
  PLOGD << "Simulation mainloop finished.";
}

template class ntt::Simulation<ntt::One_D>;
template class ntt::Simulation<ntt::Two_D>;
template class ntt::Simulation<ntt::Three_D>;

} // namespace ntt
