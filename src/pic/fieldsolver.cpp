#include "global.h"
#include "simulation.h"

#include "faraday.hpp"
#include "ampere.hpp"
#include "add_reset_currents.hpp"

#include <plog/Log.h>

namespace ntt {

// solve dB/dt
void Simulation1D::faradayHalfsubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "1D faraday";
  if (m_sim_params.m_coord_system == CARTESIAN_COORD) {
    const real_t coeff {static_cast<real_t>(0.5) * m_sim_params.m_correction
                        * m_sim_params.m_timestep / m_meshblock.get_dx1()};
    Kokkos::parallel_for(
        "faraday", m_meshblock.loopActiveCells(), Faraday1D_Cartesian(m_meshblock, coeff));
  }
}
void Simulation2D::faradayHalfsubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "2D faraday";
  if (m_sim_params.m_coord_system == CARTESIAN_COORD) {
    const real_t coeff {static_cast<real_t>(0.5) * m_sim_params.m_correction
                        * m_sim_params.m_timestep / m_meshblock.get_dx1()};
    Kokkos::parallel_for(
        "faraday", m_meshblock.loopActiveCells(), Faraday2D_Cartesian(m_meshblock, coeff));
  }
}
void Simulation3D::faradayHalfsubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "3D faraday";
  if (m_sim_params.m_coord_system == CARTESIAN_COORD) {
    const real_t coeff {static_cast<real_t>(0.5) * m_sim_params.m_correction
                        * m_sim_params.m_timestep / m_meshblock.get_dx1()};
    Kokkos::parallel_for(
        "faraday", m_meshblock.loopActiveCells(), Faraday3D_Cartesian(m_meshblock, coeff));
  }
}

// solve dE/dt
void Simulation1D::ampereSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "1D ampere";
  if (m_sim_params.m_coord_system == CARTESIAN_COORD) {
    const real_t coeff {m_sim_params.m_correction * m_sim_params.m_timestep
                        / m_meshblock.get_dx1()};
    Kokkos::parallel_for(
        "ampere", m_meshblock.loopActiveCells(), Ampere1D_Cartesian(m_meshblock, coeff));
  }
}
void Simulation2D::ampereSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "2D ampere";
  if (m_sim_params.m_coord_system == CARTESIAN_COORD) {
    const real_t coeff {m_sim_params.m_correction * m_sim_params.m_timestep
                        / m_meshblock.get_dx1()};
    Kokkos::parallel_for(
        "ampere", m_meshblock.loopActiveCells(), Ampere2D_Cartesian(m_meshblock, coeff));
  }
}
void Simulation3D::ampereSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "3D ampere";
  if (m_sim_params.m_coord_system == CARTESIAN_COORD) {
    const real_t coeff {m_sim_params.m_correction * m_sim_params.m_timestep
                        / m_meshblock.get_dx1()};
    Kokkos::parallel_for(
        "ampere", m_meshblock.loopActiveCells(), Ampere3D_Cartesian(m_meshblock, coeff));
  }
}

// add currents to E
void Simulation1D::addCurrentsSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "1D add current";
  Kokkos::parallel_for("addcurrs", m_meshblock.loopActiveCells(), AddCurrents1D(m_meshblock));
}
void Simulation2D::addCurrentsSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "2D add current";
  Kokkos::parallel_for("addcurrs", m_meshblock.loopActiveCells(), AddCurrents2D(m_meshblock));
}
void Simulation3D::addCurrentsSubstep(const real_t& time) {
  UNUSED(time);
  Kokkos::parallel_for("addcurrs", m_meshblock.loopActiveCells(), AddCurrents3D(m_meshblock));
}

// reset currents to zero
void Simulation1D::resetCurrentsSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "1D add current";
  Kokkos::parallel_for("resetcurrs", m_meshblock.loopActiveCells(), ResetCurrents1D(m_meshblock));
}
void Simulation2D::resetCurrentsSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "2D add current";
  Kokkos::parallel_for("resetcurrs", m_meshblock.loopActiveCells(), ResetCurrents2D(m_meshblock));
}
void Simulation3D::resetCurrentsSubstep(const real_t& time) {
  UNUSED(time);
  Kokkos::parallel_for("resetcurrs", m_meshblock.loopActiveCells(), ResetCurrents3D(m_meshblock));
}

} // namespace ntt
