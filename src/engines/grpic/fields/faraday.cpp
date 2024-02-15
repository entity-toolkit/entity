/**
 * @file faraday.cpp
 * @brief pushes varios version of B/B0 with E
 * @implements: `Faraday` method of the `GRPIC` class
 * @includes: `kernels/faraday_gr.hpp`
 * @depends: `grpic.h`
 *
 */

#include "wrapper.h"

#include "grpic.h"

#include "kernels/faraday_gr.hpp"

#include METRIC_HEADER

namespace ntt {

  template <>
  void GRPIC<Dim2>::Faraday(const real_t& fraction, const gr_faraday& g) {
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff  = fraction * params.correction() * mblock.timestep();
    if (g == gr_faraday::aux) {
      Kokkos::parallel_for("Farday-1",
                           mblock.rangeActiveCells(),
                           Faraday_kernel<Dim2, Metric<Dim2>>(mblock.em0,
                                                              mblock.em0,
                                                              mblock.aux,
                                                              mblock.metric,
                                                              coeff,
                                                              mblock.Ni2(),
                                                              mblock.boundaries));
    } else if (g == gr_faraday::main) {
      Kokkos::parallel_for("Farday-2",
                           mblock.rangeActiveCells(),
                           Faraday_kernel<Dim2, Metric<Dim2>>(mblock.em,
                                                              mblock.em0,
                                                              mblock.aux,
                                                              mblock.metric,
                                                              coeff,
                                                              mblock.Ni2(),
                                                              mblock.boundaries));
    } else {
      NTTHostError("Wrong option for `g`");
    }
    NTTLog();
  }

  template <>
  void GRPIC<Dim3>::Faraday(const real_t&, const gr_faraday&) {
    NTTHostError("not implemented");
  }

} // namespace ntt