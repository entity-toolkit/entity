#if SIMTYPE == GRPIC_SIMTYPE

#  include "global.h"
#  include "grpic.h"
#  include "grpic_faraday.hpp"

#  include <stdexcept>

namespace ntt {

  template <>
  void GRPIC<Dimension::TWO_D>::faradaySubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    Kokkos::parallel_for("faraday", m_mblock.loopActiveCells(), Faraday_push<Dimension::TWO_D>(m_mblock, coeff));
  }

  template <>
  void GRPIC<Dimension::THREE_D>::faradaySubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    (void)(fraction);
    (void)(coeff);
    NTTError("faraday for this metric not defined");
  }

} // namespace ntt

#endif