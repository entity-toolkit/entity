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
        auto range_p = NTT1DRange({mblock.i_max}, {mblock.i_max + N_GHOSTS});
        Kokkos::parallel_for("1d_bc_x1m", range_m, FldBC1D_PeriodicX1m(mblock, mblock.Ni));
        Kokkos::parallel_for("1d_bc_x1p", range_p, FldBC1D_PeriodicX1p(mblock, mblock.Ni));
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
        auto range_m = NTT2DRange({0, mblock.j_min}, {N_GHOSTS, mblock.j_max});
        auto range_p = NTT2DRange(
            {mblock.i_max, mblock.j_min},
            {mblock.i_max + N_GHOSTS, mblock.j_max});
        Kokkos::parallel_for("2d_bc_x1m", range_m, FldBC2D_PeriodicX1m(mblock, mblock.Ni));
        Kokkos::parallel_for("2d_bc_x1p", range_p, FldBC2D_PeriodicX1p(mblock, mblock.Ni));
      } else {
        // non-periodic
        throw std::logic_error("# 2d boundary condition NOT IMPLEMENTED.");
      }
      // corners are included in x2
      if (m_sim_params.m_boundaries[1] == PERIODIC_BC) {
        ntt_2drange_t range_m, range_p;
        if (m_sim_params.m_boundaries[0] == PERIODIC_BC) {
          // double periodic boundaries
          range_m = NTT2DRange({0, 0}, {mblock.i_max + N_GHOSTS, N_GHOSTS});
          range_p = NTT2DRange({0, mblock.j_max}, {mblock.i_max + N_GHOSTS, mblock.j_max + N_GHOSTS});
        } else {
          // single periodic (only x2-periodic)
          range_m = NTT2DRange({mblock.i_min, 0}, {mblock.i_max, N_GHOSTS});
          range_p = NTT2DRange({mblock.i_min, mblock.j_max}, {mblock.i_max, mblock.j_max + N_GHOSTS});
        }
        Kokkos::parallel_for("2d_bc_x2m", range_m, FldBC2D_PeriodicX2m(mblock, mblock.Nj));
        Kokkos::parallel_for("2d_bc_x2p", range_p, FldBC2D_PeriodicX2p(mblock, mblock.Nj));
      } else {
        // non-periodic
        throw std::logic_error("# 2d boundary condition NOT IMPLEMENTED.");
      }
    } else if ((m_sim_params.m_coord_system == "spherical") ||
               (m_sim_params.m_coord_system == "qspherical")) {
      // * * * * * * * * * * * * * * * *
      // axisymmetric spherical grid
      // * * * * * * * * * * * * * * * *
      // r = rmin boundary
      if (m_sim_params.m_boundaries[0] == USER_BC) {
        m_pGen.userBCFields(time, m_sim_params, mblock);
      } else {
        throw std::logic_error("# 2d non-user boundary condition NOT IMPLEMENTED.");
      }
      // theta = 0 boundary
      Kokkos::parallel_for(
          "2d_bc_theta0",
          NTT2DRange({0, 0}, {mblock.i_max + N_GHOSTS, mblock.j_min + 1}),
          Lambda(index_t i, index_t j) {
            mblock.em_fields(i, j, fld::bx2) = 0.0;
            mblock.em_fields(i, j, fld::ex3) = 0.0;
          });
      // theta = pi boundary
      Kokkos::parallel_for(
          "2d_bc_thetaPi",
          NTT2DRange({0, mblock.j_max}, {mblock.i_max + N_GHOSTS, mblock.j_max + N_GHOSTS}),
          Lambda(index_t i, index_t j) {
            mblock.em_fields(i, j, fld::bx2) = 0.0;
            mblock.em_fields(i, j, fld::ex3) = 0.0;
          });

      auto r_absorb {m_sim_params.m_coord_parameters[2]};
      auto r_max {mblock.grid->x1_max};
      // auto dx1 {mblock.get_dx1()};
      // auto dx2 {mblock.get_dx2()};
      Kokkos::parallel_for(
          "2d_absorbing bc",
          mblock.loopActiveCells(),
          Lambda(index_t i, index_t j) {
            auto i_ {static_cast<real_t>(i)};
            auto j_ {static_cast<real_t>(j)};

            // i
            auto [r1_, th1_] = mblock.grid->coord_CU_to_Sph(i_, ZERO);
            auto delta_r1 {(r1_ - r_absorb) / (r_max - r_absorb)};
            auto sigma_r1 {HEAVISIDE(delta_r1) * delta_r1 * delta_r1 * delta_r1};
            // i + 1/2
            auto [r2_, th2_] = mblock.grid->coord_CU_to_Sph(i_ + HALF, ZERO);
            auto delta_r2 {(r2_ - r_absorb) / (r_max - r_absorb)};
            auto sigma_r2 {HEAVISIDE(delta_r2) * delta_r2 * delta_r2 * delta_r2};

            mblock.em_fields(i, j, fld::ex1) = (1.0 - sigma_r1) * mblock.em_fields(i, j, fld::ex1);
            mblock.em_fields(i, j, fld::bx2) = (1.0 - sigma_r1) * mblock.em_fields(i, j, fld::bx2);
            mblock.em_fields(i, j, fld::bx3) = (1.0 - sigma_r1) * mblock.em_fields(i, j, fld::bx3);

            auto br_hat_target {m_pGen.userTargetField_br_HAT(mblock, i_, j_ + HALF)};
            auto bx1_source {mblock.em_fields(i, j, fld::bx1)};
            auto br_hat_source {mblock.grid->vec_CNT_to_HAT_x1(bx1_source, i_, j_ + HALF)};
            auto bx1_hat_interm {(1.0 - sigma_r2) * br_hat_source + sigma_r2 * br_hat_target};
            mblock.em_fields(i, j, fld::bx1) = mblock.grid->vec_HAT_to_CNT_x1(bx1_hat_interm, i_, j_ + HALF);
            mblock.em_fields(i, j, fld::ex2) = (1.0 - sigma_r2) * mblock.em_fields(i, j, fld::ex2);
            mblock.em_fields(i, j, fld::ex3) = (1.0 - sigma_r2) * mblock.em_fields(i, j, fld::ex3);
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
