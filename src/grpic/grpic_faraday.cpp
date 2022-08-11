#include "global.h"
#include "grpic.h"
#include "grpic_faraday.hpp"

#include <stdexcept>

namespace ntt {
  const auto Dim2 = Dimension::TWO_D;
  const auto Dim3 = Dimension::THREE_D;

  template <>
  void
  GRPIC<Dim2>::faradaySubstep(const real_t&, const real_t& fraction, const gr_faraday& f) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    if (f == gr_faraday::aux) {
      Kokkos::parallel_for(
        "faraday", m_mblock.rangeActiveCells(), FaradayGR_aux<Dim2>(m_mblock, coeff));
    } else if (f == gr_faraday::main) {
      Kokkos::parallel_for(
        "faraday", m_mblock.rangeActiveCells(), FaradayGR<Dim2>(m_mblock, coeff));
    } else {
      NTTError("Wrong option for `f`");
    }
  }

  template <>
  void GRPIC<Dim3>::faradaySubstep(const real_t&, const real_t&, const gr_faraday&) {
    NTTError("3D GRPIC not implemented yet");
  }

} // namespace ntt