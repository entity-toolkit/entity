#include "global.h"
#include "simulation.h"

#include "field_periodic_bc.hpp"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {

void Simulation1D::fieldBoundaryConditions(const real_t& time) {
  UNUSED(time);
  auto nx1 = m_meshblock.get_n1();
  if (m_sim_params.m_boundaries[0] == PERIODIC_BC) {
    auto range_m = NTT1DRange({0}, {N_GHOSTS});
    auto range_p = NTT1DRange({m_meshblock.get_imax()}, {m_meshblock.get_imax() + N_GHOSTS});
    Kokkos::parallel_for("1d_bc_x1m", range_m, FldBC1D_PeriodicX1m(m_meshblock, nx1));
    Kokkos::parallel_for("1d_bc_x1p", range_p, FldBC1D_PeriodicX1p(m_meshblock, nx1));
  } else {
    throw std::logic_error("ERROR: only periodic boundaries are implemented");
  }
}

void Simulation2D::fieldBoundaryConditions(const real_t& time) {
  UNUSED(time);
  auto nx1 = m_meshblock.get_n1();
  if (m_sim_params.m_boundaries[0] == PERIODIC_BC) {
    auto range_m = NTT2DRange({0, m_meshblock.get_jmin()}, {N_GHOSTS, m_meshblock.get_jmax()});
    auto range_p = NTT2DRange({m_meshblock.get_imax(), m_meshblock.get_jmin()},
                              {m_meshblock.get_imax() + N_GHOSTS, m_meshblock.get_jmax()});
    Kokkos::parallel_for("2d_bc_x1m", range_m, FldBC2D_PeriodicX1m(m_meshblock, nx1));
    Kokkos::parallel_for("2d_bc_x1p", range_p, FldBC2D_PeriodicX1p(m_meshblock, nx1));
  } else {
    throw std::logic_error("ERROR: only periodic boundaries are implemented");
  }
  // corners are included in x2
  auto nx2 = m_meshblock.get_n2();
  if (m_sim_params.m_boundaries[1] == PERIODIC_BC) {
    auto range_m = NTT2DRange({0, 0}, {m_meshblock.get_imax() + N_GHOSTS, N_GHOSTS});
    auto range_p
        = NTT2DRange({0, m_meshblock.get_jmax()},
                     {m_meshblock.get_imax() + N_GHOSTS, m_meshblock.get_jmax() + N_GHOSTS});
    Kokkos::parallel_for("2d_bc_x2m", range_m, FldBC2D_PeriodicX2m(m_meshblock, nx2));
    Kokkos::parallel_for("2d_bc_x2p", range_p, FldBC2D_PeriodicX2p(m_meshblock, nx2));
  } else {
    throw std::logic_error("ERROR: only periodic boundaries are implemented");
  }
}

void Simulation3D::fieldBoundaryConditions(const real_t& time) {
  UNUSED(time);
  if (m_sim_params.m_boundaries[0] == PERIODIC_BC) {

  } else {
    throw std::logic_error("ERROR: only periodic boundaries are implemented");
  }
}

} // namespace ntt
