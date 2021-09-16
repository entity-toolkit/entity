#include "global.h"
#include "picsim.h"

#include <plog/Log.h>

namespace ntt {

void PICSimulation1D::faradayHalfsubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation2D::faradayHalfsubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation3D::faradayHalfsubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation1D::ampereSubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation2D::ampereSubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation3D::ampereSubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation1D::addCurrentsSubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation2D::addCurrentsSubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation3D::addCurrentsSubstep(const real_t &time) {
  PLOGD << time;
}


}
