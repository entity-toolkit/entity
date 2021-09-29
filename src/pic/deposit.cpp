#include "global.h"
#include "simulation.h"

#include <plog/Log.h>

namespace ntt {

template <> void Simulation<One_D>::depositSubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "1D deposit";
}

template <> void Simulation<Two_D>::depositSubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "2D deposit";
}

template <> void Simulation<Three_D>::depositSubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "3D deposit";
}

template class ntt::Simulation<ntt::One_D>;
template class ntt::Simulation<ntt::Two_D>;
template class ntt::Simulation<ntt::Three_D>;

} // namespace ntt
