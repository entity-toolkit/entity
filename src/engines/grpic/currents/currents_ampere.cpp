/**
 * @file currents_ampere.cpp
 * @brief D^(n+1) = D' - 4 pi * dt * J
 * @implements: `AmpereCurrents` method of the `GRPIC` class
 * @includes: `kernels/ampere_gr.hpp`
 * @depends: `grpic.h`
 *
 * @notes: - minus sign in the current is included with the `coeff`, ...
 *           ... so the kernel adds `coeff * J`
 *         - charge renormalization is done to keep the charge density ...
 *           ... independent of the resolution and `ppc0`
 */

#include "wrapper.h"

#include "grpic.h"

#include "kernels/ampere_gr.hpp"

#include METRIC_HEADER

namespace ntt {

  /**
   * @brief Add currents to the D-field
   */
  template <>
  void GRPIC<Dim2>::AmpereCurrents(const gr_ampere& g) {
    auto& mblock = this->meshblock;
    auto  params = *(this->params());

    // const auto dt     = mblock.timestep();
    // const auto rho0   = params.larmor0();
    // const auto de0    = params.skindepth0();
    // const auto ncells = mblock.Ni1() * mblock.Ni2() * mblock.Ni3();
    // const auto rmin   = mblock.metric.getParameter("rhorizon");
    // const auto volume = constant::TWO_PI * SQR(rmin);
    // const auto n0     = params.ppc0() * (real_t)ncells / volume;
    // const auto coeff  = -dt * rho0 / (n0 * SQR(de0));
    auto range = CreateRangePolicy<Dim2>({ mblock.i1_min(), mblock.i2_min() },
                                         { mblock.i1_max(), mblock.i2_max() + 1 });

    const auto coeff = -mblock.timestep() * params.q0() / params.B0();
    if (g == gr_ampere::aux) {
      Kokkos::parallel_for(
        "AmpereCurrents-1",
        range,
        CurrentsAmpere_kernel<Dim2, Metric<Dim2>>(mblock.em0,
                                                  mblock.cur,
                                                  mblock.metric,
                                                  coeff,
                                                  mblock.Ni2(),
                                                  mblock.boundaries));
    } else if (g == gr_ampere::main) {
      Kokkos::parallel_for(
        "AmpereCurrents-2",
        range,
        CurrentsAmpere_kernel<Dim2, Metric<Dim2>>(mblock.em0,
                                                  mblock.cur0,
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
  void GRPIC<Dim3>::AmpereCurrents(const gr_ampere&) {
    NTTHostError("not implemented");
  }

} // namespace ntt
