#include "global.h"
#include "simulation.h"

#include "faraday.hpp"
#include "ampere.hpp"
#include "add_currents.hpp"

#include <plog/Log.h>

namespace ntt {

// solve dB/dt
template <Dimension D>
void Simulation<D>::faradayHalfsubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << D << "D faraday";
  const real_t coeff {static_cast<real_t>(0.5) * m_sim_params.m_correction * m_sim_params.m_timestep};
  const real_t coeff_x1 {(m_meshblock.get_dx1() != ZERO) ? coeff / m_meshblock.get_dx1() : ZERO};
  const real_t coeff_x2 {(m_meshblock.get_dx2() != ZERO) ? coeff / m_meshblock.get_dx2() : ZERO};
  const real_t coeff_x3 {(m_meshblock.get_dx3() != ZERO) ? coeff / m_meshblock.get_dx3() : ZERO};
  Kokkos::parallel_for("faraday", m_meshblock.loopActiveCells(), Faraday<D>(m_meshblock, coeff_x1, coeff_x2, coeff_x3));
}
// solve dE/dt
template <Dimension D>
void Simulation<D>::ampereSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << D << "D ampere";
  const real_t coeff {m_sim_params.m_correction * m_sim_params.m_timestep};
  const real_t coeff_x1 {(m_meshblock.get_dx1() != ZERO) ? coeff / m_meshblock.get_dx1() : ZERO};
  const real_t coeff_x2 {(m_meshblock.get_dx2() != ZERO) ? coeff / m_meshblock.get_dx2() : ZERO};
  const real_t coeff_x3 {(m_meshblock.get_dx3() != ZERO) ? coeff / m_meshblock.get_dx3() : ZERO};
  Kokkos::parallel_for("ampere", m_meshblock.loopActiveCells(), Ampere<D>(m_meshblock, coeff_x1, coeff_x2, coeff_x3));
}

// add currents to E
template <Dimension D>
void Simulation<D>::addCurrentsSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << D << "D add current";
  Kokkos::parallel_for("addcurrs", m_meshblock.loopActiveCells(), AddCurrents<D>(m_meshblock));
}

// reset currents to zero
template <Dimension D>
void Simulation<D>::resetCurrentsSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << D << "D reset current";
  Kokkos::deep_copy(m_meshblock.jx1, 0.0);
  Kokkos::deep_copy(m_meshblock.jx2, 0.0);
  Kokkos::deep_copy(m_meshblock.jx3, 0.0);
}

} // namespace ntt

template class ntt::Simulation<ntt::ONE_D>;
template class ntt::Simulation<ntt::TWO_D>;
template class ntt::Simulation<ntt::THREE_D>;
