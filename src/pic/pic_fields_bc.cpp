#include "global.h"
#include "pic.h"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {
  /**
   * @brief 1d periodic field bc.
   * 
   */
  template<>
  void PIC<Dimension::ONE_D>::fieldBoundaryConditions(const real_t&) {
    using index_t = typename RealFieldND<Dimension::ONE_D, 6>::size_type;
    if (m_mblock.metric->label == "minkowski") {
      if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
        auto range_m {NTTRange<Dimension::ONE_D>({0}, {m_mblock.i_min()})};
        auto range_p {NTTRange<Dimension::ONE_D>({m_mblock.i_max()}, {m_mblock.i_max() + N_GHOSTS})};
        auto ni {m_mblock.Ni()};
        // in x1_min
        Kokkos::parallel_for(
          "1d_bc_x1m", range_m, Lambda(index_t i) {
            m_mblock.em(i, em::ex1) = m_mblock.em(i + ni, em::ex1);
            m_mblock.em(i, em::ex2) = m_mblock.em(i + ni, em::ex2);
            m_mblock.em(i, em::ex3) = m_mblock.em(i + ni, em::ex3);
            m_mblock.em(i, em::bx1) = m_mblock.em(i + ni, em::bx1);
            m_mblock.em(i, em::bx2) = m_mblock.em(i + ni, em::bx2);
            m_mblock.em(i, em::bx3) = m_mblock.em(i + ni, em::bx3);
          });
        // in x1_max
        Kokkos::parallel_for(
          "1d_bc_x1p", range_p, Lambda(index_t i) {
            m_mblock.em(i, em::ex1) = m_mblock.em(i - ni, em::ex1);
            m_mblock.em(i, em::ex2) = m_mblock.em(i - ni, em::ex2);
            m_mblock.em(i, em::ex3) = m_mblock.em(i - ni, em::ex3);
            m_mblock.em(i, em::bx1) = m_mblock.em(i - ni, em::bx1);
            m_mblock.em(i, em::bx2) = m_mblock.em(i - ni, em::bx2);
            m_mblock.em(i, em::bx3) = m_mblock.em(i - ni, em::bx3);
          });
      } else {
        NTTError("boundary condition not implemented");
      }
    } else {
      NTTError("only minkowski possible in 1d");
    }
  }

  /**
   * @brief 2d periodic field bc.
   *
   */
  template <>
  void PIC<Dimension::TWO_D>::fieldBoundaryConditions(const real_t&) {
    using index_t = typename RealFieldND<Dimension::TWO_D, 6>::size_type;
    if (m_mblock.metric->label == "minkowski") {
      if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
        // non-periodic
        auto range_m {NTTRange<Dimension::TWO_D>({0, m_mblock.j_min()}, {m_mblock.i_min(), m_mblock.j_max()})};
        auto range_p {
          NTTRange<Dimension::TWO_D>({m_mblock.i_max(), m_mblock.j_min()}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_max()})};
        auto ni {m_mblock.Ni()};
        Kokkos::parallel_for(
          "2d_bc_x1m", range_m, Lambda(index_t i, index_t j) {
            m_mblock.em(i, j, em::ex1) = m_mblock.em(i + ni, j, em::ex1);
            m_mblock.em(i, j, em::ex2) = m_mblock.em(i + ni, j, em::ex2);
            m_mblock.em(i, j, em::ex3) = m_mblock.em(i + ni, j, em::ex3);
            m_mblock.em(i, j, em::bx1) = m_mblock.em(i + ni, j, em::bx1);
            m_mblock.em(i, j, em::bx2) = m_mblock.em(i + ni, j, em::bx2);
            m_mblock.em(i, j, em::bx3) = m_mblock.em(i + ni, j, em::bx3);
          });
        Kokkos::parallel_for(
          "2d_bc_x1p", range_p, Lambda(index_t i, index_t j) {
            m_mblock.em(i, j, em::ex1) = m_mblock.em(i - ni, j, em::ex1);
            m_mblock.em(i, j, em::ex2) = m_mblock.em(i - ni, j, em::ex2);
            m_mblock.em(i, j, em::ex3) = m_mblock.em(i - ni, j, em::ex3);
            m_mblock.em(i, j, em::bx1) = m_mblock.em(i - ni, j, em::bx1);
            m_mblock.em(i, j, em::bx2) = m_mblock.em(i - ni, j, em::bx2);
            m_mblock.em(i, j, em::bx3) = m_mblock.em(i - ni, j, em::bx3);
          });
      } else {
        // non-periodic
        NTTError("2d boundary condition for minkowski not implemented");
      }
      // corners are included in x2
      if (m_mblock.boundaries[1] == BoundaryCondition::PERIODIC) {
        RangeND<Dimension::TWO_D> range_m, range_p;
        if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
          // double periodic boundaries
          range_m = NTTRange<Dimension::TWO_D>({0, 0}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_min()});
          range_p = NTTRange<Dimension::TWO_D>({0, m_mblock.j_max()}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_max() + N_GHOSTS});
        } else {
          // single periodic (only x2-periodic)
          range_m = NTTRange<Dimension::TWO_D>({m_mblock.i_min(), 0}, {m_mblock.i_max(), m_mblock.j_min()});
          range_p = NTTRange<Dimension::TWO_D>({m_mblock.i_min(), m_mblock.j_max()}, {m_mblock.i_max(), m_mblock.j_max() + N_GHOSTS});
        }
        auto nj {m_mblock.Nj()};
        Kokkos::parallel_for(
          "2d_bc_x2m", range_m, Lambda(index_t i, index_t j) {
            m_mblock.em(i, j, em::ex1) = m_mblock.em(i, j + nj, em::ex1);
            m_mblock.em(i, j, em::ex2) = m_mblock.em(i, j + nj, em::ex2);
            m_mblock.em(i, j, em::ex3) = m_mblock.em(i, j + nj, em::ex3);
            m_mblock.em(i, j, em::bx1) = m_mblock.em(i, j + nj, em::bx1);
            m_mblock.em(i, j, em::bx2) = m_mblock.em(i, j + nj, em::bx2);
            m_mblock.em(i, j, em::bx3) = m_mblock.em(i, j + nj, em::bx3);
          });
        Kokkos::parallel_for(
          "2d_bc_x2p", range_p, Lambda(index_t i, index_t j) {
            m_mblock.em(i, j, em::ex1) = m_mblock.em(i, j - nj, em::ex1);
            m_mblock.em(i, j, em::ex2) = m_mblock.em(i, j - nj, em::ex2);
            m_mblock.em(i, j, em::ex3) = m_mblock.em(i, j - nj, em::ex3);
            m_mblock.em(i, j, em::bx1) = m_mblock.em(i, j - nj, em::bx1);
            m_mblock.em(i, j, em::bx2) = m_mblock.em(i, j - nj, em::bx2);
            m_mblock.em(i, j, em::bx3) = m_mblock.em(i, j - nj, em::bx3);
          });
      } else {
        // non-periodic
        NTTError("2d boundary condition for minkowski not implemented");
      }
    // } else if ((m_sim_params.m_coord_system == "spherical") || (m_sim_params.m_coord_system == "qspherical")) {
    //   // * * * * * * * * * * * * * * * *
    //   // axisymmetric spherical grid
    //   // * * * * * * * * * * * * * * * *
    //   // r = rmin boundary
    //   if (m_sim_params.m_boundaries[0] == USER_BC) {
    //     m_pGen.userBCFields(time, m_sim_params, mblock);
    //   } else {
    //     throw std::logic_error("# 2d non-user boundary condition NOT IMPLEMENTED.");
    //   }
    //   // theta = 0 boundary
    //   Kokkos::parallel_for(
    //       "2d_bc_theta0",
    //       NTT2DRange({0, 0}, {mblock.i_max + N_GHOSTS, mblock.j_min + 1}),
    //       Lambda(index_t i, index_t j) {
    //         mblock.em_fields(i, j, fld::bx2) = 0.0;
    //         mblock.em_fields(i, j, fld::ex3) = 0.0;
    //       });
    //   // theta = pi boundary
    //   Kokkos::parallel_for(
    //       "2d_bc_thetaPi",
    //       NTT2DRange({0, mblock.j_max}, {mblock.i_max + N_GHOSTS, mblock.j_max + N_GHOSTS}),
    //       Lambda(index_t i, index_t j) {
    //         mblock.em_fields(i, j, fld::bx2) = 0.0;
    //         mblock.em_fields(i, j, fld::ex3) = 0.0;
    //       });

    //   auto r_absorb {m_sim_params.m_coord_parameters[2]};
    //   auto r_max {mblock.grid->x1_max};
    //   // auto dx1 {mblock.get_dx1()};
    //   // auto dx2 {mblock.get_dx2()};
    //   Kokkos::parallel_for(
    //       "2d_absorbing bc",
    //       mblock.loopActiveCells(),
    //       Lambda(index_t i, index_t j) {
    //         auto i_ {static_cast<real_t>(i)};
    //         auto j_ {static_cast<real_t>(j)};

    //         // i
    //         auto [r1_, th1_] = mblock.grid->coord_CU_to_Sph(i_, ZERO);
    //         auto delta_r1 {(r1_ - r_absorb) / (r_max - r_absorb)};
    //         auto sigma_r1 {HEAVISIDE(delta_r1) * delta_r1 * delta_r1 * delta_r1};
    //         // i + 1/2
    //         auto [r2_, th2_] = mblock.grid->coord_CU_to_Sph(i_ + HALF, ZERO);
    //         auto delta_r2 {(r2_ - r_absorb) / (r_max - r_absorb)};
    //         auto sigma_r2 {HEAVISIDE(delta_r2) * delta_r2 * delta_r2 * delta_r2};

    //         mblock.em_fields(i, j, fld::ex1) = (1.0 - sigma_r1) * mblock.em_fields(i, j, fld::ex1);
    //         mblock.em_fields(i, j, fld::bx2) = (1.0 - sigma_r1) * mblock.em_fields(i, j, fld::bx2);
    //         mblock.em_fields(i, j, fld::bx3) = (1.0 - sigma_r1) * mblock.em_fields(i, j, fld::bx3);

    //         auto br_hat_target {m_pGen.userTargetField_br_HAT(mblock, i_, j_ + HALF)};
    //         auto bx1_source {mblock.em_fields(i, j, fld::bx1)};
    //         auto br_hat_source {mblock.grid->vec_CNT_to_HAT_x1(bx1_source, i_, j_ + HALF)};
    //         auto bx1_hat_interm {(1.0 - sigma_r2) * br_hat_source + sigma_r2 * br_hat_target};
    //         mblock.em_fields(i, j, fld::bx1) = mblock.grid->vec_HAT_to_CNT_x1(bx1_hat_interm, i_, j_ + HALF);
    //         mblock.em_fields(i, j, fld::ex2) = (1.0 - sigma_r2) * mblock.em_fields(i, j, fld::ex2);
    //         mblock.em_fields(i, j, fld::ex3) = (1.0 - sigma_r2) * mblock.em_fields(i, j, fld::ex3);
    //       });
    } else {
      NTTError("2d boundary condition for metric not implemented");
    }
  }

  /**
   * @brief 3d periodic field bc.
   *
   */
  template <>
  void PIC<Dimension::THREE_D>::fieldBoundaryConditions(const real_t&) {
    NTTError("not implemented");
  }

} // namespace ntt
