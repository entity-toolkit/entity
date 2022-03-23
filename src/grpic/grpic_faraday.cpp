#include "global.h"
#include "grpic.h"
#include "grpic_faraday.hpp"

#include <stdexcept>

namespace ntt {

  template <>
  void GRPIC<Dimension::TWO_D>::faradaySubstep(const real_t&, const real_t& fraction, const gr_faraday& f) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    if (f == gr_faraday::aux) {
      Kokkos::parallel_for("faraday", m_mblock.loopActiveCells(), FaradayGR_aux<Dimension::TWO_D>(m_mblock, coeff));
    } else if (f == gr_faraday::main) {
      Kokkos::parallel_for("faraday", m_mblock.loopActiveCells(), FaradayGR<Dimension::TWO_D>(m_mblock, coeff));
    } else {
      NTTError("Wrong option for `f`");
    }
  }

  template <>
  void GRPIC<Dimension::THREE_D>::faradaySubstep(const real_t&, const real_t&, const gr_faraday&) {
    NTTError("3D GRPIC not implemented yet");
  }

} // namespace ntt