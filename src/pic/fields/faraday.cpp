#include "global.h"
#include "pic.h"

#include <plog/Log.h>

#ifdef MINKOWSKI_METRIC
#  include "faraday_mink.hpp"
#else
#  include "faraday_curv.hpp"
#endif

#include <stdexcept>

namespace ntt {
  template <>
  void PIC<Dim1>::Faraday(const real_t& fraction) {
#ifdef MINKOWSKI_METRIC
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff {fraction * params.correction() * mblock.timestep()};
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(mblock.metric.x1_max - mblock.metric.x1_min) / mblock.metric.nx1};
    Kokkos::parallel_for(
      "faraday", mblock.rangeActiveCells(), Faraday_kernel<Dim1>(mblock, coeff / dx));
#else
    (void)(fraction);
    NTTHostError("faraday for this metric not defined");
#endif
    PLOGD << "... ... faraday substep finished";
  }

  template <>
  void PIC<Dim2>::Faraday(const real_t& fraction) {
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff {fraction * params.correction() * mblock.timestep()};
#ifdef MINKOWSKI_METRIC
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(mblock.metric.x1_max - mblock.metric.x1_min) / mblock.metric.nx1};
    Kokkos::parallel_for(
      "faraday", mblock.rangeActiveCells(), Faraday_kernel<Dim2>(mblock, coeff / dx));
#else
    Kokkos::parallel_for(
      "faraday", mblock.rangeActiveCells(), Faraday_kernel<Dim2>(mblock, coeff));
#endif
    PLOGD << "... ... faraday substep finished";
  }

  template <>
  void PIC<Dim3>::Faraday(const real_t& fraction) {
#ifdef MINKOWSKI_METRIC
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff {fraction * params.correction() * mblock.timestep()};
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(mblock.metric.x1_max - mblock.metric.x1_min) / mblock.metric.nx1};
    Kokkos::parallel_for(
      "faraday", mblock.rangeActiveCells(), Faraday_kernel<Dim3>(mblock, coeff / dx));
#else
    (void)(fraction);
    NTTHostError("faraday for this metric not defined");
#endif
    PLOGD << "... ... faraday substep finished";
  }

} // namespace ntt