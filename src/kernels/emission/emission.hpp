#ifndef KERNELS_EMISSION_EMISSION_HPP
#define KERNELS_EMISSION_EMISSION_HPP

#include "enums.h"

#include "traits/metric.h"

namespace kernel {
  using namespace ntt;

  template <SimEngine S, MetricClass M>
  struct NoEmissionPolicy_t {};

} // namespace kernel

#endif
