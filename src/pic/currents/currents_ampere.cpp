#include "currents_ampere.hpp"

#include "wrapper.h"

#include "pic.h"

namespace ntt {
  /**
   * @brief Add currents to the E-field
   */
  template <Dimension D>
  void PIC<D>::AmpereCurrents() {
    auto&      mblock = this->meshblock;
    auto       params = *(this->params());
    const auto dt     = mblock.timestep();
    const auto rho0   = params.larmor0();
    const auto de0    = params.skindepth0();
    const auto ncells = mblock.Ni1() * mblock.Ni2() * mblock.Ni3();
#ifdef MINKOWSKI_METRIC
    const auto sx1    = (short)D > 0 ? (mblock.metric.x1_max - mblock.metric.x1_min) : 1.0;
    const auto sx2    = (short)D > 1 ? (mblock.metric.x2_max - mblock.metric.x2_min) : 1.0;
    const auto sx3    = (short)D > 2 ? (mblock.metric.x3_max - mblock.metric.x3_min) : 1.0;
    const auto volume = sx1 * sx2 * sx3;
    const auto n0     = params.ppc0() * (real_t)ncells / volume;
    const auto dx     = (mblock.metric.x1_max - mblock.metric.x1_min) / mblock.metric.nx1;
    const auto coeff  = -dt * rho0 * math::pow(dx, (3 - (int)D))
                       / (n0 * SQR(de0) * mblock.metric.sqrt_det_h({ ZERO }));
    Kokkos::parallel_for(
      "add_currents", mblock.rangeActiveCells(), CurrentsAmpere_kernel<D>(mblock, coeff));
#else
    const auto                       rmin   = mblock.metric.x1_min;
    const auto                       volume = constant::TWO_PI * SQR(rmin);
    const auto                       n0     = params.ppc0() * (real_t)ncells / volume;
    const auto                       coeff  = -dt * rho0 / (n0 * SQR(de0));

    tuple_t<tuple_t<short, Dim2>, D> range;
    // skip the axis
    if constexpr (D == Dim1) {
      range[0][0] = 0;
      range[0][1] = 0;
    } else if constexpr (D == Dim2) {
      range[0][0] = 0;
      range[0][1] = 0;
      range[1][0] = 1;
      range[1][1] = -1;
    } else if constexpr (D == Dim3) {
      range[0][0] = 0;
      range[0][1] = 0;
      range[1][0] = 1;
      range[1][1] = -1;
      range[2][0] = 0;
      range[2][1] = 0;
    }
    Kokkos::parallel_for(
      "add_currents", mblock.rangeCells(range), CurrentsAmpere_kernel<D>(mblock, coeff));
    // do axes separately
    if constexpr (D == Dim2) {
      Kokkos::parallel_for("add_currents_pole",
                           CreateRangePolicy<Dim1>({ mblock.i1_min() }, { mblock.i1_max() }),
                           CurrentsAmperePoles_kernel<Dim2>(mblock, coeff));
    }
#endif
    PLOGD << "... ... ampere currents substep finished";
  }
}    // namespace ntt

template void ntt::PIC<ntt::Dim1>::AmpereCurrents();
template void ntt::PIC<ntt::Dim2>::AmpereCurrents();
template void ntt::PIC<ntt::Dim3>::AmpereCurrents();