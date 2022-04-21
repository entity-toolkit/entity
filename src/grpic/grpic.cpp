#include "global.h"
#include "timer.h"
#include "grpic.h"

namespace ntt {

  template <Dimension D>
  void GRPIC<D>::mainloop() {}

  template <Dimension D>
  void GRPIC<D>::process() {}

  template <Dimension D>
  void GRPIC<D>::step_forward(const real_t&) {}

  template <Dimension D>
  void GRPIC<D>::step_backward(const real_t&) {}
} // namespace ntt

#if SIMTYPE == GRPIC_SIMTYPE
template class ntt::GRPIC<ntt::Dimension::TWO_D>;
template class ntt::GRPIC<ntt::Dimension::THREE_D>;
#endif