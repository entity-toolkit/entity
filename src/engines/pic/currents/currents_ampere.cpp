/**
 * @file currents_ampere.cpp
 * @brief E^(n+1) = E' - 4 pi * dt * J
 * @implements: `AmpereCurrents` method of the `PIC` class
 * @includes: `kernels/ampere_mink.hpp` or `kernels/ampere_sr.hpp`
 * @depends: `pic.h`
 *
 * @notes: - minus sign in the current is included with the `coeff`, ...
 *           ... so the kernel adds `coeff * J`
 *         - charge renormalization is done to keep the charge density ...
 *           ... independent of the resolution and `ppc0`
 */

#include "wrapper.h"

#include "pic.h"

#ifdef MINKOWSKI_METRIC
  #include "kernels/ampere_mink.hpp"
#else
  #include "kernels/ampere_sr.hpp"
#endif

#include METRIC_HEADER

namespace ntt {

#ifdef MINKOWSKI_METRIC

  /**
   * @brief Add currents to the E-field
   */
  template <Dimension D>
  void PIC<D>::AmpereCurrents() {
    auto&      mblock = this->meshblock;
    auto       params = *(this->params());
    const auto coeff  = -mblock.timestep() * params.q0() * params.n0() /
                       (params.B0() * params.V0());
    Kokkos::parallel_for(
      "AmpereCurrents",
      mblock.rangeActiveCells(),
      CurrentsAmpere_kernel<D>(mblock.em, mblock.cur, coeff, ONE / params.n0()));

    NTTLog();
  }
#else

  /**
   * @brief Add currents to the E-field
   */
  template <>
  void PIC<Dim2>::AmpereCurrents() {
    auto&      mblock = this->meshblock;
    auto       params = *(this->params());
    const auto coeff  = -mblock.timestep() * params.q0() * params.n0() /
                       params.B0();

    Kokkos::parallel_for(
      "AmpereCurrents",
      CreateRangePolicy<Dim2>({ mblock.i1_min(), mblock.i2_min() },
                              { mblock.i1_max(), mblock.i2_max() + 1 }),
      CurrentsAmpere_kernel<Dim2, Metric<Dim2>>(mblock.em,
                                                mblock.cur,
                                                mblock.metric,
                                                coeff,
                                                ONE / params.n0(),
                                                mblock.Ni2(),
                                                mblock.boundaries));
    NTTLog();
  }

  template <>
  void PIC<Dim1>::AmpereCurrents() {
    NTTHostError("not applicable");
  }

  template <>
  void PIC<Dim3>::AmpereCurrents() {
    NTTHostError("not implemented");
  }

#endif
} // namespace ntt

#ifdef MINKOWSKI_METRIC
template void ntt::PIC<ntt::Dim1>::AmpereCurrents();
template void ntt::PIC<ntt::Dim2>::AmpereCurrents();
template void ntt::PIC<ntt::Dim3>::AmpereCurrents();
#endif
