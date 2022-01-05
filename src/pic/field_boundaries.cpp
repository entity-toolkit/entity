#include "global.h"
#include "simulation.h"

#include "cartesian.h"
#include "spherical.h"

#include "field_periodic_bc.hpp"

#include <plog/Log.h>

#include <stdexcept>
#include <iostream>

namespace ntt {

  // # # # # # # # # # # # # # # # #
  // 1d
  // # # # # # # # # # # # # # # # #
  template <>
  void Simulation<ONE_D>::fieldBoundaryConditions(const real_t& time) {
    UNUSED(time);
    if (m_sim_params.m_coord_system == "cartesian") {
      if (m_sim_params.m_boundaries[0] == PERIODIC_BC) {
        auto range_m = NTT1DRange({0}, {N_GHOSTS});
        auto range_p = NTT1DRange({m_meshblock.i_max}, {m_meshblock.i_max + N_GHOSTS});
        Kokkos::parallel_for("1d_bc_x1m", range_m, FldBC1D_PeriodicX1m(m_meshblock, m_meshblock.Ni));
        Kokkos::parallel_for("1d_bc_x1p", range_p, FldBC1D_PeriodicX1p(m_meshblock, m_meshblock.Ni));
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
    using index_t = NTTArray<real_t**>::size_type;
    if (m_sim_params.m_coord_system == "cartesian") {
      // * * * * * * * * * * * *
      // cartesian grid
      // * * * * * * * * * * * *
      if (m_sim_params.m_boundaries[0] == PERIODIC_BC) {
        auto range_m = NTT2DRange({0, m_meshblock.j_min}, {N_GHOSTS, m_meshblock.j_max});
        auto range_p = NTT2DRange(
            {m_meshblock.i_max, m_meshblock.j_min},
            {m_meshblock.i_max + N_GHOSTS, m_meshblock.j_max});
        Kokkos::parallel_for("2d_bc_x1m", range_m, FldBC2D_PeriodicX1m(m_meshblock, m_meshblock.Ni));
        Kokkos::parallel_for("2d_bc_x1p", range_p, FldBC2D_PeriodicX1p(m_meshblock, m_meshblock.Ni));
      } else {
        // non-periodic
        throw std::logic_error("# 2d boundary condition NOT IMPLEMENTED.");
      }
      // corners are included in x2
      if (m_sim_params.m_boundaries[1] == PERIODIC_BC) {
        ntt_2drange_t range_m, range_p;
        if (m_sim_params.m_boundaries[0] == PERIODIC_BC) {
          // double periodic boundaries
          range_m = NTT2DRange({0, 0}, {m_meshblock.i_max + N_GHOSTS, N_GHOSTS});
          range_p = NTT2DRange({0, m_meshblock.j_max}, {m_meshblock.i_max + N_GHOSTS, m_meshblock.j_max + N_GHOSTS});
        } else {
          // single periodic (only x2-periodic)
          range_m = NTT2DRange({m_meshblock.i_min, 0}, {m_meshblock.i_max, N_GHOSTS});
          range_p = NTT2DRange({m_meshblock.i_min, m_meshblock.j_max}, {m_meshblock.i_max, m_meshblock.j_max + N_GHOSTS});
        }
        Kokkos::parallel_for("2d_bc_x2m", range_m, FldBC2D_PeriodicX2m(m_meshblock, m_meshblock.Nj));
        Kokkos::parallel_for("2d_bc_x2p", range_p, FldBC2D_PeriodicX2p(m_meshblock, m_meshblock.Nj));
      } else {
        // non-periodic
        throw std::logic_error("# 2d boundary condition NOT IMPLEMENTED.");
      }
    // } else if ((m_sim_params.m_coord_system == "spherical") ||
    //            (m_sim_params.m_coord_system == "qspherical")) {
    //   // * * * * * * * * * * * * * * * *
    //   // axisymmetric spherical grid
    //   // * * * * * * * * * * * * * * * *
    //   if (m_sim_params.m_boundaries[0] == USER_BC) {
    //     m_pGen.userBCFields(time, m_sim_params, m_meshblock);
    //   } else {
    //     throw std::logic_error("# 2d non-user boundary condition NOT IMPLEMENTED.");
    //   }
    //   // theta = 0 boundary
    //   Kokkos::parallel_for(
    //       "2d_bc_theta0",
    //       NTT2DRange({0, 0}, {m_meshblock.get_imax() + N_GHOSTS, m_meshblock.get_jmin() + 1}),
    //       Lambda(index_t i, index_t j) {
    //         m_meshblock.em_fields(i, j, fld::bx2) = 0.0;
    //         m_meshblock.em_fields(i, j, fld::ex3) = 0.0;
    //       });
    //   // theta = pi boundary
    //   Kokkos::parallel_for(
    //       "2d_bc_thetaPi",
    //       NTT2DRange({0, m_meshblock.get_jmax()}, {m_meshblock.get_imax() + N_GHOSTS, m_meshblock.get_jmax() + N_GHOSTS}),
    //       Lambda(index_t i, index_t j) {
    //         m_meshblock.em_fields(i, j, fld::bx2) = 0.0;
    //         m_meshblock.em_fields(i, j, fld::ex3) = 0.0;
    //       });

    //   auto r_absorb {m_sim_params.m_coord_parameters[2]};
    //   auto r_max {m_meshblock.m_coord_system->getSpherical_r(m_meshblock.m_extent[1], ZERO)};
    //   auto dx1 {m_meshblock.get_dx1()};
    //   auto dx2 {m_meshblock.get_dx2()};
    //   Kokkos::parallel_for(
    //       "2d_absorbing bc",
    //       m_meshblock.loopActiveCells(),
    //       Lambda(index_t i, index_t j) {
    //         auto x1 {m_meshblock.convert_iTOx1(i)};
    //         auto x2 {m_meshblock.convert_jTOx2(j)};
    //         real_t r, delta_r, sigma_r;
    //         real_t bx1_target {m_pGen.userTargetField_bx1(m_meshblock, x1, x2 + 0.5 * dx2)};
    //         r = m_meshblock.m_coord_system->getSpherical_r(x1 + 0.5 * dx1, ZERO);
    //         delta_r = (r - r_absorb) / (r_max - r_absorb);
    //         sigma_r = HEAVISIDE(delta_r) * delta_r * delta_r * delta_r;
    //         m_meshblock.em_fields(i, j, fld::ex1) = (1.0 - sigma_r) * m_meshblock.em_fields(i, j, fld::ex1);
    //         m_meshblock.em_fields(i, j, fld::bx2) = (1.0 - sigma_r) * m_meshblock.em_fields(i, j, fld::bx2);

    //         m_meshblock.em_fields(i, j, fld::bx3) = (1.0 - sigma_r) * m_meshblock.em_fields(i, j, fld::bx3);

    //         r = m_meshblock.m_coord_system->getSpherical_r(x1, ZERO);
    //         delta_r = (r - r_absorb) / (r_max - r_absorb);
    //         sigma_r = HEAVISIDE(delta_r) * delta_r * delta_r * delta_r;
    //         m_meshblock.em_fields(i, j, fld::bx1) = (1.0 - sigma_r) * m_meshblock.em_fields(i, j, fld::bx1) + sigma_r * bx1_target;
    //         m_meshblock.em_fields(i, j, fld::ex2) = (1.0 - sigma_r) * m_meshblock.em_fields(i, j, fld::ex2);
    //         m_meshblock.em_fields(i, j, fld::ex3) = (1.0 - sigma_r) * m_meshblock.em_fields(i, j, fld::ex3);
    //       });
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
