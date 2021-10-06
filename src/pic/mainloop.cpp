#include "global.h"
#include "timer.h"
#include "simulation.h"

#include <plog/Log.h>

namespace ntt {

template <template <typename T> class D>
void Simulation<D>::step_forward(const real_t& time) {
  TimerCollection timers({"Field_Solver", "Field_BC", "Curr_Deposit", "Prtl_Pusher"});
  {
    timers.start(1);
    faradayHalfsubstep(time);
    timers.stop(1);
  }

  {
    timers.start(2);
    fieldBoundaryConditions(time);
    timers.stop(2);
  }

  // particlePushSubstep(time);

  // depositSubstep(time);

  // BC particles
  // BC currents

  {
    timers.start(1);
    faradayHalfsubstep(time);
    timers.stop(1);
  }

  {
    timers.start(2);
    fieldBoundaryConditions(time);
    timers.stop(2);
  }

  {
    timers.start(1);
    ampereSubstep(time);
    addCurrentsSubstep(time);
    resetCurrentsSubstep(time);
    timers.stop(1);
  }

  {
    timers.start(2);
    fieldBoundaryConditions(time);
    timers.stop(2);
  }
  timers.printAll(millisecond);
}

template <template <typename T> class D>
void Simulation<D>::mainloop() {
  PLOGD << "Simulation mainloop started.";

  unsigned long timax{static_cast<unsigned long>(m_sim_params.m_runtime / m_sim_params.m_timestep)};
  real_t time{0.0};
  for (unsigned long ti{0}; ti < timax; ++ti) {
    PLOGD << "t = " << time;
    step_forward(time);
    time += m_sim_params.m_timestep;
  }
  PLOGD << "Simulation mainloop finished.";
}

template class ntt::Simulation<ntt::One_D>;
template class ntt::Simulation<ntt::Two_D>;
template class ntt::Simulation<ntt::Three_D>;

} // namespace ntt
