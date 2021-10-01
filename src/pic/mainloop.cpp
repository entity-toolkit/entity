#include "global.h"
#include "timer.h"
#include "simulation.h"

#include <plog/Log.h>

namespace ntt {

template <template <typename T> class D> void Simulation<D>::mainloop() {
  PLOGD << "Simulation mainloop started.";
  TimerCollection timers({"Field_Solver", "Curr_Deposit", "Prtl_Pusher"});
  unsigned long timax{static_cast<unsigned long>(m_sim_params.m_runtime / m_sim_params.m_timestep)};
  real_t time{0.0};
  for (unsigned long ti{0}; ti < timax; ++ti) {
    PLOGD << "t = " << time;
    timers.start(1);
    faradayHalfsubstep(time);
    timers.stop(1);
    // BC b-fields

    // particlePushSubstep(time);

    // depositSubstep(time);

    // BC particles
    // BC currents

    timers.start(1);
    faradayHalfsubstep(time);
    ampereSubstep(time);
    addCurrentsSubstep(time);
    resetCurrentsSubstep(time);
    timers.stop(1);

    timers.printAll(millisecond);
    time += m_sim_params.m_timestep;
  }
  PLOGD << "Simulation mainloop finished.";
}

template class ntt::Simulation<ntt::One_D>;
template class ntt::Simulation<ntt::Two_D>;
template class ntt::Simulation<ntt::Three_D>;

} // namespace ntt
