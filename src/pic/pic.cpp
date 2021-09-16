#include "global.h"
#include "picsim.h"

namespace ntt {

void PICSimulation::stepForward(const real_t &time) {
  timer_em.start();
  faradayHalfsubstep(time);
  timer_em.stop();

  // BC b-fields

  timer_pusher.start();
  particlePushSubstep(time);
  timer_pusher.stop();

  timer_deposit.start();
  depositSubstep(time);
  timer_deposit.stop();

  // BC particles
  // BC currents

  timer_em.start();
  faradayHalfsubstep(time);
  ampereSubstep(time);
  addCurrentsSubstep(time);
  timer_em.stop();

  // BC e,b-fields
}

}
