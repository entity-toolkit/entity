#ifndef KERNELS_EMISSION_EMISSION_HPP
#define KERNELS_EMISSION_EMISSION_HPP

#include "enums.h"

namespace kernel {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct NoEmissionPolicy_t {};

} // namespace kernel

#endif
