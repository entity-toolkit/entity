/**
 * @file faraday.cpp
 * @brief B^n = B^n-1/2 - (dt/2) * curl E^(n)
 * @implements: `Faraday` method of the `PIC` class
 * @includes: `faraday_mink.hpp` or `faraday_curv.hpp`
 * @depends: `pic.h`
 *
 * @notes: - `dx` (cell size) is passed to the solver explicitly ...
 *           ... in minkowski case to avoid trivial metric computations.
 *
 */

#include "wrapper.h"

#include "pic.h"

#include <plog/Log.h>

#ifdef MINKOWSKI_METRIC
#  include "faraday_mink.hpp"
#else
#  include "faraday_curv.hpp"
#endif

#include <stdexcept>

namespace ntt {

#ifdef MINKOWSKI_METRIC
  template <Dimension D>
  void PIC<D>::Faraday(const real_t& fraction) {
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff { fraction * params.correction() * mblock.timestep() };
    const auto   dx { (mblock.metric.x1_max - mblock.metric.x1_min) / mblock.metric.nx1 };
    Kokkos::parallel_for(
      "faraday", mblock.rangeActiveCells(), Faraday_kernel<D>(mblock, coeff / dx));
    PLOGD << "... ... faraday substep finished";
  }

#else

  template <>
  void PIC<Dim2>::Faraday(const real_t& fraction) {
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff { fraction * params.correction() * mblock.timestep() };
    Kokkos::parallel_for(
      "faraday", mblock.rangeActiveCells(), Faraday_kernel<Dim2>(mblock, coeff));
    PLOGD << "... ... faraday substep finished";
  }

  template <>
  void PIC<Dim1>::Faraday(const real_t&) {
    NTTHostError("not applicable");
  }
  template <>
  void PIC<Dim3>::Faraday(const real_t&) {
    NTTHostError("not implemented");
  }

#endif

}    // namespace ntt

template void ntt::PIC<ntt::Dim1>::Faraday(const real_t&);
template void ntt::PIC<ntt::Dim2>::Faraday(const real_t&);
template void ntt::PIC<ntt::Dim3>::Faraday(const real_t&);