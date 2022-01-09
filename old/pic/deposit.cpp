#include "global.h"
#include "simulation.h"

#include <plog/Log.h>

namespace ntt {

  template <Dimension D>
  void Simulation<D>::depositSubstep(const real_t& time) {
    UNUSED(time);
    PLOGD << D << "D deposit";
  }

  // template<>
  // void Simulation<ONE_D>::depositSubstep(const real_t& time) {
  //   UNUSED(time);
  //   PLOGD << "1D deposit";
  // }
  //
  // void Simulation<TWO_D>::depositSubstep(const real_t& time) {
  //   UNUSED(time);
  //   PLOGD << "2D deposit";
  // }
  //
  // void Simulation3D::depositSubstep(const real_t& time) {
  //   UNUSED(time);
  //   PLOGD << "3D deposit";
  // }

} // namespace ntt

template class ntt::Simulation<ntt::ONE_D>;
template class ntt::Simulation<ntt::TWO_D>;
template class ntt::Simulation<ntt::THREE_D>;
