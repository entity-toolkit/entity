#include "global.h"
#include "simulation.h"

#include "cartesian.h"
#include "spherical.h"

#include "field_periodic_bc.hpp"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {

  // # # # # # # # # # # # # # # # #
  // 1d
  // # # # # # # # # # # # # # # # #
  template <>
  void Simulation<ONE_D>::fieldBoundaryConditions(const real_t& time) {
    UNUSED(time);
    if (m_sim_params.m_coord_system == "cartesian") {
      auto nx1 = m_meshblock.get_n1();
      if (m_sim_params.m_boundaries[0] == PERIODIC_BC) {
        auto range_m = NTT1DRange({0}, {N_GHOSTS});
        auto range_p = NTT1DRange({m_meshblock.get_imax()}, {m_meshblock.get_imax() + N_GHOSTS});
        Kokkos::parallel_for("1d_bc_x1m", range_m, FldBC1D_PeriodicX1m(m_meshblock, nx1));
        Kokkos::parallel_for("1d_bc_x1p", range_p, FldBC1D_PeriodicX1p(m_meshblock, nx1));
      } else {
        // non-periodic
        throw std::logic_error("# 1d boundary condition NOT IMPLEMENTED.");
      }
    } else {
      throw std::logic_error("# Error: only a Cartesian system is possible in 1d.");
    }
  }

  // # # # # # # # # # # # # # # # #
  // 2d
  // # # # # # # # # # # # # # # # #
  template <>
  void Simulation<TWO_D>::fieldBoundaryConditions(const real_t& time) {
    UNUSED(time);
    using index_t = NTTArray<real_t**>::size_type;
    if (m_sim_params.m_coord_system == "cartesian") {
      // * * * * * * * * * * * *
      // cartesian grid
      // * * * * * * * * * * * *
      auto nx1 = m_meshblock.get_n1();
      if (m_sim_params.m_boundaries[0] == PERIODIC_BC) {
        auto range_m = NTT2DRange({0, m_meshblock.get_jmin()}, {N_GHOSTS, m_meshblock.get_jmax()});
        auto range_p = NTT2DRange(
            {m_meshblock.get_imax(), m_meshblock.get_jmin()},
            {m_meshblock.get_imax() + N_GHOSTS, m_meshblock.get_jmax()});
        Kokkos::parallel_for("2d_bc_x1m", range_m, FldBC2D_PeriodicX1m(m_meshblock, nx1));
        Kokkos::parallel_for("2d_bc_x1p", range_p, FldBC2D_PeriodicX1p(m_meshblock, nx1));
      } else {
        // non-periodic
        throw std::logic_error("# 2d boundary condition NOT IMPLEMENTED.");
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
          range_m = NTT2DRange({m_meshblock.get_imin(), 0}, {m_meshblock.get_imax(), N_GHOSTS});
          range_p = NTT2DRange({m_meshblock.get_imin(), m_meshblock.get_jmax()}, {m_meshblock.get_imax(), m_meshblock.get_jmax() + N_GHOSTS});
        }
        Kokkos::parallel_for("2d_bc_x2m", range_m, FldBC2D_PeriodicX2m(m_meshblock, nx2));
        Kokkos::parallel_for("2d_bc_x2p", range_p, FldBC2D_PeriodicX2p(m_meshblock, nx2));
      } else {
        // non-periodic
        throw std::logic_error("# 2d boundary condition NOT IMPLEMENTED.");
      }
    } else if ((m_sim_params.m_coord_system == "spherical") || (m_sim_params.m_coord_system == "qspherical")) {
      // * * * * * * * * * * * *
      // axisymmetric spherical grid
      // * * * * * * * * * * * *
      // rmin boundary
      if (m_sim_params.m_boundaries[0] == USER_BC) {
        m_pGen.userBCFields_x1min(m_sim_params, m_meshblock);
      } else {
        throw std::logic_error("# 2d non-user boundary condition NOT IMPLEMENTED.");
      }
      // rmax boundary
      if (m_sim_params.m_boundaries[1] == USER_BC) {
        m_pGen.userBCFields_x1max(m_sim_params, m_meshblock);
      } else {
        throw std::logic_error("# 2d non-user boundary condition NOT IMPLEMENTED.");
      }
      // theta = 0 boundary
      Kokkos::parallel_for(
          "2d_bc_theta0",
          NTT2DRange({0, 0}, {m_meshblock.get_imax() + N_GHOSTS, m_meshblock.get_jmin() + 1}),
          Lambda(index_t i, index_t j) {
            m_meshblock.em_fields(i, j, fld::bx2) = 0.0;
            m_meshblock.em_fields(i, j, fld::ex3) = 0.0;
          });
      // theta = pi boundary
      Kokkos::parallel_for(
          "2d_bc_thetaPi",
          NTT2DRange({0, m_meshblock.get_jmax()}, {m_meshblock.get_imax() + N_GHOSTS, m_meshblock.get_jmax() + N_GHOSTS}),
          Lambda(index_t i, index_t j) {
            m_meshblock.em_fields(i, j, fld::bx2) = 0.0;
            m_meshblock.em_fields(i, j, fld::ex3) = 0.0;
          });
    } else {
      throw std::logic_error("# 2d boundary condition for coordinate system NOT IMPLEMENTED.");
    }
  }

  // # # # # # # # # # # # # # # # #
  // 3d
  // # # # # # # # # # # # # # # # #
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
