#if SIMTYPE == GRPIC_SIMTYPE

#  include "global.h"
#  include "grpic.h"
#  include "grpic_ampere.hpp"

#  include <stdexcept>

namespace ntt {

  template <>
  void GRPIC<Dimension::TWO_D>::ampereSubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    Kokkos::parallel_for("ampere",
      NTTRange<Dimension::TWO_D>({m_mblock.i_min(), m_mblock.j_min() + 1}, {m_mblock.i_max(), m_mblock.j_max()}),
      Ampere_push<Dimension::TWO_D>(m_mblock, coeff));
    Kokkos::parallel_for("ampere_pole",
                         NTTRange<Dimension::ONE_D>({m_mblock.i_min()}, {m_mblock.i_max()}),
                         Ampere_Poles<Dimension::TWO_D>(m_mblock, coeff));
  }

  template <>
  void GRPIC<Dimension::THREE_D>::ampereSubstep(const real_t&, const real_t& fraction) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    (void)(fraction);
    (void)(coeff);
    NTTError("ampere for this metric not defined");
  }

} // namespace ntt


#endif