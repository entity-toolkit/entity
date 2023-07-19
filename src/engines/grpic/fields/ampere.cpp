#include "ampere.hpp"

#include "wrapper.h"

#include "grpic.h"

namespace ntt {

  template <>
  void GRPIC<Dim2>::Ampere(const real_t& fraction, const gr_ampere& g) {
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());

    const real_t coeff  = fraction * params.correction() * mblock.timestep();
    auto         range  = CreateRangePolicy<Dim2>({ mblock.i1_min(), mblock.i2_min() + 1 },
                                         { mblock.i1_max(), mblock.i2_max() });
    auto range_pole { CreateRangePolicy<Dim1>({ mblock.i1_min() }, { mblock.i1_max() }) };
    if (g == gr_ampere::aux) {
      Kokkos::parallel_for("Ampere-1", range, AmpereAux_kernel<Dim2>(mblock, coeff));
      Kokkos::parallel_for("Ampere-2", range_pole, AmpereAuxPoles_kernel<Dim2>(mblock, coeff));
    } else if (g == gr_ampere::main) {
      Kokkos::parallel_for("Ampere-3", range, Ampere_kernel<Dim2>(mblock, coeff));
      Kokkos::parallel_for("Ampere-4", range_pole, AmperePoles_kernel<Dim2>(mblock, coeff));
    } else if (g == gr_ampere::init) {
      Kokkos::parallel_for("Ampere-5", range, AmpereInit_kernel<Dim2>(mblock, coeff));
      Kokkos::parallel_for("Ampere-6", range_pole, AmpereInitPoles_kernel<Dim2>(mblock, coeff));
    } else {
      NTTHostError("Wrong option for `g`");
    }
    NTTLog();
  }

  template <>
  void GRPIC<Dim3>::Ampere(const real_t&, const gr_ampere&) {
    NTTHostError("not implemented");
  }

}    // namespace ntt