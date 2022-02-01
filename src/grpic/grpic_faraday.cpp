#  include "global.h"
#  include "grpic.h"
#  include "grpic_faraday.hpp"

#  include <stdexcept>

namespace ntt {

  template <>
  void GRPIC<Dimension::TWO_D>::faradaySubstep(const real_t&, const real_t& fraction, const short (&s)) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    if (s == 0) {
    Kokkos::parallel_for("faraday", m_mblock.loopActiveCells(), Faraday_push0<Dimension::TWO_D>(m_mblock, coeff));
    } else if (s == 1) {
    Kokkos::parallel_for("faraday", m_mblock.loopActiveCells(), Faraday_push<Dimension::TWO_D>(m_mblock, coeff));
    } else {
    NTTError("Only two options: 0 and 1");
    }
  }

  template <>
  void GRPIC<Dimension::THREE_D>::faradaySubstep(const real_t&, const real_t& fraction, const short (&s)) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    (void)(fraction);
    (void)(coeff);
    NTTError("faraday for this metric not defined");
  }

} // namespace ntt