#include "global.h"
#include "grpic.h"
#include "grpic_ampere.hpp"

#include <stdexcept>

namespace ntt {

  template <>
  void GRPIC<Dim2>::ampereSubstep(const real_t&, const real_t& fraction, const gr_ampere& f) {
    const real_t coeff {fraction * m_sim_params.correction() * m_mblock.timestep()};
    auto         range {
      CreateRangePolicy<Dim2>({m_mblock.i1_min(), m_mblock.i2_min() + 1}, {m_mblock.i1_max(), m_mblock.i2_max()})};
    auto range_pole {CreateRangePolicy<Dim1>({m_mblock.i1_min()}, {m_mblock.i1_max()})};
    if (f == gr_ampere::aux) {
      Kokkos::parallel_for("ampere", range, AmpereGR_aux<Dim2>(m_mblock, coeff));
      Kokkos::parallel_for("ampere_pole", range_pole, AmperePolesGR_aux<Dim2>(m_mblock, coeff));
    } else if (f == gr_ampere::main) {
      Kokkos::parallel_for("ampere", range, AmpereGR<Dim2>(m_mblock, coeff));
      Kokkos::parallel_for("ampere_pole", range_pole, AmperePolesGR<Dim2>(m_mblock, coeff));
    } else if (f == gr_ampere::init) {
      Kokkos::parallel_for("ampere", range, AmpereGR_init<Dim2>(m_mblock, coeff));
      Kokkos::parallel_for("ampere_pole", range_pole, AmperePolesGR_init<Dim2>(m_mblock, coeff));
    } else {
      NTTError("Wrong option for `f`");
    }
  }

  template <>
  void GRPIC<Dim3>::ampereSubstep(const real_t&, const real_t&, const gr_ampere&) {
    NTTError("3D GRPIC not implemented yet");
  }

} // namespace ntt