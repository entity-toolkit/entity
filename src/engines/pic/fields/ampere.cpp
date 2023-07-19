/**
 * @file ampere.cpp
 * @brief E' = E^n + dt * curl B^(n+1/2)
 * @implements: `Ampere` method of the `PIC` class
 * @includes: `ampere_mink.hpp` or `ampere_curv.hpp`
 * @depends: `pic.h`
 *
 * @notes: - `dx` (cell size) is passed to the solver explicitly ...
 *           ... in minkowski case to avoid trivial metric computations.
 *
 */

#include "wrapper.h"

#include "pic.h"

#ifdef MINKOWSKI_METRIC
#  include "ampere_mink.hpp"
#else
#  include "ampere_curv.hpp"
#endif

#include <stdexcept>

namespace ntt {

#ifdef MINKOWSKI_METRIC
  template <Dimension D>
  void PIC<D>::Ampere(const real_t& fraction) {
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff { fraction * params.correction() * mblock.timestep() };
    const auto   dx { (mblock.metric.x1_max - mblock.metric.x1_min) / mblock.metric.nx1 };
    Kokkos::parallel_for(
      "Ampere", mblock.rangeActiveCells(), Ampere_kernel<D>(mblock, coeff / dx));
    NTTLog();
  }

#else

  template <>
  void PIC<Dim2>::Ampere(const real_t& fraction) {
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff { fraction * params.correction() * mblock.timestep() };
    Kokkos::parallel_for("Ampere",
                         CreateRangePolicy<Dim2>({ mblock.i1_min(), mblock.i2_min() + 1 },
                                                 { mblock.i1_max(), mblock.i2_max() }),
                         Ampere_kernel<Dim2>(mblock, coeff));
    Kokkos::parallel_for("Ampere-1",
                         CreateRangePolicy<Dim1>({ mblock.i1_min() }, { mblock.i1_max() }),
                         AmperePoles_kernel<Dim2>(mblock, coeff));
    NTTLog();
  }

  template <>
  void PIC<Dim1>::Ampere(const real_t&) {
    NTTHostError("not applicable");
  }
  template <>
  void PIC<Dim3>::Ampere(const real_t&) {
    NTTHostError("not implemented");
  }

#endif

}    // namespace ntt

#ifdef MINKOWSKI_METRIC
template void ntt::PIC<ntt::Dim1>::Ampere(const real_t&);
template void ntt::PIC<ntt::Dim2>::Ampere(const real_t&);
template void ntt::PIC<ntt::Dim3>::Ampere(const real_t&);
#endif