#include "global.h"
#include "picsim.h"

#include <plog/Log.h>

namespace ntt {

void PICSimulation::particlePushSubstep(const real_t &time) {
  PLOGD << time;
}

}
