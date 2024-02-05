/**
 * @file ampere.cpp
 * @brief pushes varios version of D/D0 with H
 * @implements: `Ampere` method of the `GRPIC` class
 * @includes: `kernels/ampere_gr.hpp`
 * @depends: `grpic.h`
 *
 */

#include "wrapper.h"

#include "grpic.h"

#include "kernels/ampere_gr.hpp"

#include METRIC_HEADER

namespace ntt {

  template <>
  void GRPIC<Dim2>::Ampere(const real_t& fraction, const gr_ampere& g) {
    auto& mblock = this->meshblock;
    auto  params = *(this->params());

    const real_t coeff = fraction * params.correction() * mblock.timestep();
    auto range = CreateRangePolicy<Dim2>({ mblock.i1_min(), mblock.i2_min() },
                                         { mblock.i1_max(), mblock.i2_max() + 1 });
    if (g == gr_ampere::aux) {
      // push D0 with H
      Kokkos::parallel_for("Ampere-1",
                           range,
                           Ampere_kernel<Dim2, Metric<Dim2>>(mblock.em0,
                                                             mblock.em0,
                                                             mblock.aux,
                                                             mblock.metric,
                                                             coeff,
                                                             mblock.Ni2(),
                                                             mblock.boundaries));
    } else if (g == gr_ampere::main) {
      // push D with H but assign to D0
      Kokkos::parallel_for("Ampere-2",
                           range,
                           Ampere_kernel<Dim2, Metric<Dim2>>(mblock.em,
                                                             mblock.em0,
                                                             mblock.aux,
                                                             mblock.metric,
                                                             coeff,
                                                             mblock.Ni2(),
                                                             mblock.boundaries));
    } else if (g == gr_ampere::init) {
      // push D0 with H but assign to D
      Kokkos::parallel_for("Ampere-3",
                           range,
                           Ampere_kernel<Dim2, Metric<Dim2>>(mblock.em,
                                                             mblock.em,
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
  void GRPIC<Dim3>::Ampere(const real_t&, const gr_ampere&) {
    NTTHostError("not implemented");
  }

} // namespace ntt