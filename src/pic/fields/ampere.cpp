#include "global.h"
#include "pic.h"

#ifdef MINKOWSKI_METRIC
#  include "ampere_mink.hpp"
#else
#  include "ampere_curv.hpp"
#endif

#include <stdexcept>

namespace ntt {
  template <>
  void PIC<Dim1>::Ampere(const real_t& fraction) {
#ifdef MINKOWSKI_METRIC
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff {fraction * params.correction() * mblock.timestep()};
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(mblock.metric.x1_max - mblock.metric.x1_min) / mblock.metric.nx1};
    Kokkos::parallel_for(
      "ampere", mblock.rangeActiveCells(), Ampere_kernel<Dim1>(mblock, coeff / dx));
#else
    (void)(fraction);
    NTTHostError("ampere for this metric not defined");
#endif
  }

  template <>
  void PIC<Dim2>::Ampere(const real_t& fraction) {
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff {fraction * params.correction() * mblock.timestep()};
#ifdef MINKOWSKI_METRIC
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(mblock.metric.x1_max - mblock.metric.x1_min) / mblock.metric.nx1};
    Kokkos::parallel_for(
      "ampere", mblock.rangeActiveCells(), Ampere_kernel<Dim2>(mblock, coeff / dx));
#else
    Kokkos::parallel_for("ampere",
                         CreateRangePolicy<Dim2>({mblock.i1_min(), mblock.i2_min() + 1},
                                                 {mblock.i1_max(), mblock.i2_max()}),
                         Ampere_kernel<Dim2>(mblock, coeff));
    Kokkos::parallel_for("ampere_pole",
                         CreateRangePolicy<Dim1>({mblock.i1_min()}, {mblock.i1_max()}),
                         AmperePoles_kernel<Dim2>(mblock, coeff));
#endif
  }

  template <>
  void PIC<Dim3>::Ampere(const real_t& fraction) {
#ifdef MINKOWSKI_METRIC
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff {fraction * params.correction() * mblock.timestep()};
    // dx is passed only in minkowski case to avoid trivial metric computations.
    const auto dx {(mblock.metric.x1_max - mblock.metric.x1_min) / mblock.metric.nx1};
    Kokkos::parallel_for(
      "ampere", mblock.rangeActiveCells(), Ampere_kernel<Dim3>(mblock, coeff / dx));
#else
    (void)(fraction);
    NTTHostError("ampere for this metric not defined");
#endif
  }

} // namespace ntt