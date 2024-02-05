/**
 * @file faraday.cpp
 * @brief B^n = B^n-1/2 - (dt/2) * curl E^(n)
 * @implements: `Faraday` method of the `PIC` class
 * @includes: `kernels/faraday_mink.hpp` or `kernels/faraday_sr.hpp`
 * @depends: `pic.h`
 *
 * @notes: - `dx` (cell size) is passed to the solver explicitly ...
 *           ... in minkowski case to avoid trivial metric computations.
 */

#include "wrapper.h"

#include "pic.h"

#ifdef MINKOWSKI_METRIC
  #include "kernels/faraday_mink.hpp"
#else
  #include "kernels/faraday_sr.hpp"
#endif

#include METRIC_HEADER

#include <stdexcept>

namespace ntt {

#ifdef MINKOWSKI_METRIC
  template <Dimension D>
  void PIC<D>::Faraday(const real_t& fraction) {
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff { fraction * params.correction() * mblock.timestep() };
    const auto   dx { (mblock.metric.x1_max - mblock.metric.x1_min) /
                    mblock.metric.nx1 };
    Kokkos::parallel_for("Faraday",
                         mblock.rangeActiveCells(),
                         Faraday_kernel<D>(mblock.em, coeff / dx));
    NTTLog();
  }

#else

  template <>
  void PIC<Dim2>::Faraday(const real_t& fraction) {
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff { fraction * params.correction() * mblock.timestep() };
    Kokkos::parallel_for("Faraday",
                         mblock.rangeActiveCells(),
                         Faraday_kernel<Dim2, Metric<Dim2>>(mblock.em,
                                                            mblock.metric,
                                                            coeff,
                                                            mblock.boundaries));
    NTTLog();
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

} // namespace ntt

#ifdef MINKOWSKI_METRIC
template void ntt::PIC<ntt::Dim1>::Faraday(const real_t&);
template void ntt::PIC<ntt::Dim2>::Faraday(const real_t&);
template void ntt::PIC<ntt::Dim3>::Faraday(const real_t&);
#endif