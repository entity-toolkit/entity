#include "global.h"
#include "picsim.h"

#include <plog/Log.h>

namespace ntt {

void PICSimulation1D::depositSubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation2D::depositSubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation3D::depositSubstep(const real_t &time) {
  PLOGD << time;
}

}
