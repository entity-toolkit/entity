#include "global.h"
#include "simulation.h"

#include "field_periodic_bc.hpp"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {

// 1d
template <>
void Simulation<ONE_D>::fieldBoundaryConditions(const real_t& time) {
  UNUSED(time);
  auto nx1 = m_meshblock.get_n1();
  if (m_sim_params.m_boundaries[0] == PERIODIC_BC) {
    auto range_m = NTT1DRange({0}, {N_GHOSTS});
    auto range_p = NTT1DRange({m_meshblock.get_imax()}, {m_meshblock.get_imax() + N_GHOSTS});
    Kokkos::parallel_for("1d_bc_x1m", range_m, FldBC1D_PeriodicX1m(m_meshblock, nx1));
    Kokkos::parallel_for("1d_bc_x1p", range_p, FldBC1D_PeriodicX1p(m_meshblock, nx1));
  } else {
    throw std::logic_error("# NOT IMPLEMENTED.");
  }
}

// 2d
template <>
void Simulation<TWO_D>::fieldBoundaryConditions(const real_t& time) {
  UNUSED(time);
  // if ((m_sim_params.m_coord_system == CARTESIAN_COORD) || (m_sim_params.m_coord_system == CARTESIAN_LIKE_COORD)) {
    // cartesian-like grid
    auto nx1 = m_meshblock.get_n1();
    if (m_sim_params.m_boundaries[0] == PERIODIC_BC) {
      auto range_m = NTT2DRange({0, m_meshblock.get_jmin()}, {N_GHOSTS, m_meshblock.get_jmax()});
      auto range_p = NTT2DRange(
          {m_meshblock.get_imax(), m_meshblock.get_jmin()},
          {m_meshblock.get_imax() + N_GHOSTS, m_meshblock.get_jmax()});
      Kokkos::parallel_for("2d_bc_x1m", range_m, FldBC2D_PeriodicX1m(m_meshblock, nx1));
      Kokkos::parallel_for("2d_bc_x1p", range_p, FldBC2D_PeriodicX1p(m_meshblock, nx1));
    } else {
      throw std::logic_error("# NOT IMPLEMENTED.");
    }
    // corners are included in x2
    auto nx2 = m_meshblock.get_n2();
    if (m_sim_params.m_boundaries[1] == PERIODIC_BC) {
      ntt_2drange_t range_m, range_p;
      if (m_sim_params.m_boundaries[0] == PERIODIC_BC) {
        // double periodic boundaries
        range_m = NTT2DRange({0, 0}, {m_meshblock.get_imax() + N_GHOSTS, N_GHOSTS});
        range_p = NTT2DRange({0, m_meshblock.get_jmax()}, {m_meshblock.get_imax() + N_GHOSTS, m_meshblock.get_jmax() + N_GHOSTS});
      } else {
        // single periodic (only x2-periodic)
        range_m = NTT2DRange({N_GHOSTS, 0}, {m_meshblock.get_imax(), N_GHOSTS});
        range_p = NTT2DRange({N_GHOSTS, m_meshblock.get_jmax()}, {m_meshblock.get_imax(), m_meshblock.get_jmax() + N_GHOSTS});
      }
      Kokkos::parallel_for("2d_bc_x2m", range_m, FldBC2D_PeriodicX2m(m_meshblock, nx2));
      Kokkos::parallel_for("2d_bc_x2p", range_p, FldBC2D_PeriodicX2p(m_meshblock, nx2));
    } else {
      throw std::logic_error("# NOT IMPLEMENTED.");
    }
  // } else if ((m_sim_params.m_coord_system == SPHERICAL_COORD) || (m_sim_params.m_coord_system == SPHERICAL_LIKE_COORD)) {
  //   // axisymmetric-like grid
  //
  // }
}

// 3d
template <>
void Simulation<THREE_D>::fieldBoundaryConditions(const real_t& time) {
  UNUSED(time);
  if (m_sim_params.m_boundaries[0] == PERIODIC_BC) {

  } else {
    throw std::logic_error("# NOT IMPLEMENTED.");
  }
}

} // namespace ntt

template class ntt::Simulation<ntt::ONE_D>;
template class ntt::Simulation<ntt::TWO_D>;
template class ntt::Simulation<ntt::THREE_D>;

/*
// this is what 1D periodic boundaries do:
//
//    ghosts                  reals                     ghosts
//  | _ | _ || _ | _ | _ | _ | _ | _ | _ | _ | _ | _ || _ | _ |
//    ^   ^    ⌄   ⌄                           ⌄   ⌄    ^   ^
//     \___\____\___\_________________________/___/    /   /
//               \___\________________________________/___/
//
// this is what 2D periodic boundaries do:
//
//    ghosts                  reals                     ghosts
//  | _ | _ || _ | _ | _ | _ | _ | _ | _ | _ | _ | _ || _ | _ | <-\
//  | _ | _ || _ | _ | _ | _ | _ | _ | _ | _ | _ | _ || _ | _ | <--\
//  | = | = || = = = = = = = = = = = = = = = = = = = || = | = |     |
//  | _ | _ || _ | _ | _ | _ | _ | _ | _ | _ | _ | _ || _ | _ | >---|--\
//  | _ | _ || _ | _ | _ | _ | _ | _ | _ | _ | _ | _ || _ | _ | >---|---\
//  | _ | _ || _ | _ | _ | _ | _ | _ | _ | _ | _ | _ || _ | _ |     |    |
//  | _ | _ || _ | _ | _ | _ | _ | _ | _ | _ | _ | _ || _ | _ |     |    |
//  | _ | _ || _ | _ | _ | _ | _ | _ | _ | _ | _ | _ || _ | _ |     |    |
//  | _ | _ || _ | _ | _ | _ | _ | _ | _ | _ | _ | _ || _ | _ |     |    |
//  | _ | _ || _ | _ | _ | _ | _ | _ | _ | _ | _ | _ || _ | _ | >--/     |
//  | _ | _ || _ | _ | _ | _ | _ | _ | _ | _ | _ | _ || _ | _ | >-/      |
//  | = | = || = = = = = = = = = = = = = = = = = = = || = | = |          |
//  | _ | _ || _ | _ | _ | _ | _ | _ | _ | _ | _ | _ || _ | _ | <-------/
//  | _ | _ || _ | _ | _ | _ | _ | _ | _ | _ | _ | _ || _ | _ | <------/
//    ^   ^    ⌄   ⌄                           ⌄   ⌄    ^   ^
//     \___\____\___\_________________________/___/    /   /
//               \___\________________________________/___/
//
*/
