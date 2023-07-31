#include "currents_ampere.hpp"

#include "wrapper.h"

#include "grpic.h"

#include "io/output.h"

namespace ntt {

  /**
   * @brief Add currents to the E-field
   */
  template <Dimension D>
  void GRPIC<D>::AmpereCurrents(const gr_ampere& g) {
    auto&      mblock = this->meshblock;
    auto       params = *(this->params());

    const auto dt     = mblock.timestep();
    const auto rho0   = params.larmor0();
    const auto de0    = params.skindepth0();
    const auto ncells = mblock.Ni1() * mblock.Ni2() * mblock.Ni3();
    const auto rmin   = mblock.metric.getParameter("rh");
    const auto volume = constant::TWO_PI * SQR(rmin);
    const auto n0     = params.ppc0() * (real_t)ncells / volume;
    const auto coeff  = -dt * rho0 / (n0 * SQR(de0));

    range_t<D> range;
    if constexpr (D == Dim2) {
      range = CreateRangePolicy<Dim2>({ mblock.i1_min(), mblock.i2_min() + 1 },
                                      { mblock.i1_max(), mblock.i2_max() });
    } else if constexpr (D == Dim3) {
      range
        = CreateRangePolicy<Dim3>({ mblock.i1_min(), mblock.i2_min() + 1, mblock.i3_min() },
                                  { mblock.i1_max(), mblock.i2_max(), mblock.i3_max() });
    }
    auto range_pole { CreateRangePolicy<Dim1>({ mblock.i1_min() }, { mblock.i1_max() }) };

    if (g == gr_ampere::aux) {
      // D0(n-1/2) -> (J(n)) -> D0(n+1/2)
      Kokkos::parallel_for(
        "AmpereCurrentsAux-1", range, CurrentsAmpereAux_kernel<D>(mblock, coeff));
      if constexpr (D == Dim2) {
        Kokkos::parallel_for("AmpereCurrentsAux-2",
                             range_pole,
                             CurrentsAmpereAuxPoles_kernel<Dim2>(mblock, coeff));
      }
    } else if (g == gr_ampere::main) {
      // D0(n) -> (J0(n+1/2)) -> D0(n+1)
      Kokkos::parallel_for(
        "AmpereCurrentsMain-1", range, CurrentsAmpere_kernel<D>(mblock, coeff));
      if constexpr (D == Dim2) {
        Kokkos::parallel_for(
          "AmpereCurrentsMain-2", range_pole, CurrentsAmperePoles_kernel<Dim2>(mblock, coeff));
      }
    }
    NTTLog();
  }

}    // namespace ntt

template void ntt::GRPIC<ntt::Dim2>::AmpereCurrents(const gr_ampere&);
template void ntt::GRPIC<ntt::Dim3>::AmpereCurrents(const gr_ampere&);