#include "global.h"
#include "simulation.h"

#include "faraday.hpp"
#include "ampere.hpp"
#include "add_reset_currents.hpp"

#include <plog/Log.h>

namespace ntt {

// solve dB/dt
template <>
void Simulation<One_D>::faradayHalfsubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "1D faraday";
  if (m_sim_params.m_coord_system == CARTESIAN_COORD) {
    const real_t coeff{static_cast<real_t>(0.5) * m_sim_params.m_correction
                       * m_sim_params.m_timestep / m_meshblock.get_dx1()};
    Kokkos::parallel_for(
        "faraday", loopActiveCells(m_meshblock), Faraday1D_Cartesian(m_meshblock, coeff));
  }
}
template <>
void Simulation<Two_D>::faradayHalfsubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "2D faraday";
  if (m_sim_params.m_coord_system == CARTESIAN_COORD) {
    const real_t coeff{static_cast<real_t>(0.5) * m_sim_params.m_correction
                       * m_sim_params.m_timestep / m_meshblock.get_dx1()};
    Kokkos::parallel_for(
        "faraday", loopActiveCells(m_meshblock), Faraday2D_Cartesian(m_meshblock, coeff));
  }
}
template <>
void Simulation<Three_D>::faradayHalfsubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "3D faraday";
  if (m_sim_params.m_coord_system == CARTESIAN_COORD) {
    const real_t coeff{static_cast<real_t>(0.5) * m_sim_params.m_correction
                       * m_sim_params.m_timestep / m_meshblock.get_dx1()};
    Kokkos::parallel_for(
        "faraday", loopActiveCells(m_meshblock), Faraday3D_Cartesian(m_meshblock, coeff));
  }
}

// solve dE/dt
template <>
void Simulation<One_D>::ampereSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "1D ampere";
  if (m_sim_params.m_coord_system == CARTESIAN_COORD) {
    const real_t coeff{m_sim_params.m_correction * m_sim_params.m_timestep / m_meshblock.get_dx1()};
    Kokkos::parallel_for(
        "faraday", loopActiveCells(m_meshblock), Ampere1D_Cartesian(m_meshblock, coeff));
  }
}
template <>
void Simulation<Two_D>::ampereSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "2D ampere";
  if (m_sim_params.m_coord_system == CARTESIAN_COORD) {
    const real_t coeff{m_sim_params.m_correction * m_sim_params.m_timestep / m_meshblock.get_dx1()};
    Kokkos::parallel_for(
        "faraday", loopActiveCells(m_meshblock), Ampere2D_Cartesian(m_meshblock, coeff));
  }
}
template <>
void Simulation<Three_D>::ampereSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "3D ampere";
  if (m_sim_params.m_coord_system == CARTESIAN_COORD) {
    const real_t coeff{m_sim_params.m_correction * m_sim_params.m_timestep / m_meshblock.get_dx1()};
    Kokkos::parallel_for(
        "faraday", loopActiveCells(m_meshblock), Ampere3D_Cartesian(m_meshblock, coeff));
  }
}

// add currents to E
template <>
void Simulation<One_D>::addCurrentsSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "1D add current";
  Kokkos::parallel_for("faraday", loopActiveCells(m_meshblock), AddCurrents1D(m_meshblock));
}
template <>
void Simulation<Two_D>::addCurrentsSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "2D add current";
  Kokkos::parallel_for("faraday", loopActiveCells(m_meshblock), AddCurrents2D(m_meshblock));
}
template <>
void Simulation<Three_D>::addCurrentsSubstep(const real_t& time) {
  UNUSED(time);
  Kokkos::parallel_for("faraday", loopActiveCells(m_meshblock), AddCurrents3D(m_meshblock));
}

// reset currents to zero
template <>
void Simulation<One_D>::resetCurrentsSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "1D add current";
  Kokkos::parallel_for("faraday", loopActiveCells(m_meshblock), ResetCurrents1D(m_meshblock));
}
template <>
void Simulation<Two_D>::resetCurrentsSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "2D add current";
  Kokkos::parallel_for("faraday", loopActiveCells(m_meshblock), ResetCurrents2D(m_meshblock));
}
template <>
void Simulation<Three_D>::resetCurrentsSubstep(const real_t& time) {
  UNUSED(time);
  Kokkos::parallel_for("faraday", loopActiveCells(m_meshblock), ResetCurrents3D(m_meshblock));
}

template class ntt::Simulation<ntt::One_D>;
template class ntt::Simulation<ntt::Two_D>;
template class ntt::Simulation<ntt::Three_D>;

} // namespace ntt
