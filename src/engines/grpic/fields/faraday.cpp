#include "faraday.hpp"

#include "wrapper.h"

#include "grpic.h"

namespace ntt {
  template <>
  void GRPIC<Dim2>::Faraday(const real_t& fraction, const gr_faraday& g) {
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t coeff  = fraction * params.correction() * mblock.timestep();
    if (g == gr_faraday::aux) {
      Kokkos::parallel_for("faraday",
                           mblock.rangeActiveCells(),
                           FaradayAux_kernel<Dim2>(mblock, coeff));
    } else if (g == gr_faraday::main) {
      Kokkos::parallel_for("faraday",
                           mblock.rangeActiveCells(),
                           Faraday_kernel<Dim2>(mblock, coeff));
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