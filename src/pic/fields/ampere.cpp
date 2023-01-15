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
  template <>
  void PIC<D>::Ampere(const real_t& fraction) {
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff { fraction * params.correction() * mblock.timestep() };
    const auto   dx { (mblock.metric.x1_max - mblock.metric.x1_min) / mblock.metric.nx1 };
    Kokkos::parallel_for(
      "ampere", mblock.rangeActiveCells(), Ampere_kernel<D>(mblock, coeff / dx));
    PLOGD << "... ... ampere substep finished";
  }

#else

  template <>
  void PIC<Dim2>::Ampere(const real_t& fraction) {
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff { fraction * params.correction() * mblock.timestep() };
    Kokkos::parallel_for("ampere",
                         CreateRangePolicy<Dim2>({ mblock.i1_min(), mblock.i2_min() + 1 },
                                                 { mblock.i1_max(), mblock.i2_max() }),
                         Ampere_kernel<Dim2>(mblock, coeff));
    Kokkos::parallel_for("ampere_pole",
                         CreateRangePolicy<Dim1>({ mblock.i1_min() }, { mblock.i1_max() }),
                         AmperePoles_kernel<Dim2>(mblock, coeff));
    PLOGD << "... ... ampere substep finished";
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

template void ntt::PIC<ntt::Dim1>::Ampere(const real_t&);
template void ntt::PIC<ntt::Dim2>::Ampere(const real_t&);
template void ntt::PIC<ntt::Dim3>::Ampere(const real_t&);