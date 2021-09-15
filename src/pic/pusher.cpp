#include "global.h"
#include "picsim.h"

#include <iostream>

namespace ntt {

void PICSimulation::particlePushSubstep(const real_t &time) {
  std::cout << stringifyDimension(m_domain.m_dimension) << " pusher " << time << "\n";
}

}
