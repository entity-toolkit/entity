#include "global.h"
#include "pic.h"
#include "pic_fields_bc.hpp"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {
  const auto Dim1 = Dimension::ONE_D;
  const auto Dim2 = Dimension::TWO_D;
  const auto Dim3 = Dimension::THREE_D;

  /**
   * @brief 1d periodic field bc.
   *
   */
  template <>
  void PIC<Dim1>::fieldBoundaryConditions(const real_t&) {

#if (METRIC == MINKOWSKI_METRIC)
    if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      auto mblock {this->m_mblock};
      auto range_m {NTTRange<Dim1>({0}, {m_mblock.i1_min()})};
      auto range_p {NTTRange<Dim1>({m_mblock.i1_max()}, {m_mblock.i1_max() + N_GHOSTS})};
      auto ni {m_mblock.Ni1()};
      // in x1_min
      Kokkos::parallel_for(
        "1d_bc_x1m", range_m, Lambda(index_t i) {
          mblock.em(i, em::ex1) = mblock.em(i + ni, em::ex1);
          mblock.em(i, em::ex2) = mblock.em(i + ni, em::ex2);
          mblock.em(i, em::ex3) = mblock.em(i + ni, em::ex3);
          mblock.em(i, em::bx1) = mblock.em(i + ni, em::bx1);
          mblock.em(i, em::bx2) = mblock.em(i + ni, em::bx2);
          mblock.em(i, em::bx3) = mblock.em(i + ni, em::bx3);
        });
      // in x1_max
      Kokkos::parallel_for(
        "1d_bc_x1p", range_p, Lambda(index_t i) {
          mblock.em(i, em::ex1) = mblock.em(i - ni, em::ex1);
          mblock.em(i, em::ex2) = mblock.em(i - ni, em::ex2);
          mblock.em(i, em::ex3) = mblock.em(i - ni, em::ex3);
          mblock.em(i, em::bx1) = mblock.em(i - ni, em::bx1);
          mblock.em(i, em::bx2) = mblock.em(i - ni, em::bx2);
          mblock.em(i, em::bx3) = mblock.em(i - ni, em::bx3);
        });
    } else {
      NTTError("boundary condition not implemented");
    }
#else
    (void)(index_t {});
    NTTError("only minkowski possible in 1d");
#endif
  }

  /**
   * @brief 2d periodic field bc.
   *
   */
  template <>
  void PIC<Dim2>::fieldBoundaryConditions(const real_t&) {

#if (METRIC == MINKOWSKI_METRIC)
    if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      // periodic
      auto range_m {
        NTTRange<Dim2>({0, m_mblock.i2_min()}, {m_mblock.i1_min(), m_mblock.i2_max()})};
      auto range_p {NTTRange<Dim2>({m_mblock.i1_max(), m_mblock.i2_min()},
                                   {m_mblock.i1_max() + N_GHOSTS, m_mblock.i2_max()})};
      auto ni {m_mblock.Ni1()};
      auto mblock {this->m_mblock};
      Kokkos::parallel_for(
        "2d_bc_x1m", range_m, Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i + ni, j, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i + ni, j, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i + ni, j, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i + ni, j, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i + ni, j, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i + ni, j, em::bx3);
        });
      Kokkos::parallel_for(
        "2d_bc_x1p", range_p, Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i - ni, j, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i - ni, j, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i - ni, j, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i - ni, j, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i - ni, j, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i - ni, j, em::bx3);
        });
    } else {
      // non-periodic
      NTTError("2d boundary condition for minkowski not implemented");
    }
    // corners are included in x2
    if (m_mblock.boundaries[1] == BoundaryCondition::PERIODIC) {
      RangeND<Dim2> range_m, range_p;
      if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
        // double periodic boundaries
        range_m = NTTRange<Dim2>({0, 0}, {m_mblock.i1_max() + N_GHOSTS, m_mblock.i2_min()});
        range_p = NTTRange<Dim2>({0, m_mblock.i2_max()},
                                 {m_mblock.i1_max() + N_GHOSTS, m_mblock.i2_max() + N_GHOSTS});
      } else {
        // single periodic (only x2-periodic)
        range_m
          = NTTRange<Dim2>({m_mblock.i1_min(), 0}, {m_mblock.i1_max(), m_mblock.i2_min()});
        range_p = NTTRange<Dim2>({m_mblock.i1_min(), m_mblock.i2_max()},
                                 {m_mblock.i1_max(), m_mblock.i2_max() + N_GHOSTS});
      }
      auto nj {m_mblock.Ni2()};
      auto mblock {this->m_mblock};
      Kokkos::parallel_for(
        "2d_bc_x2m", range_m, Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i, j + nj, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i, j + nj, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i, j + nj, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i, j + nj, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i, j + nj, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i, j + nj, em::bx3);
        });
      Kokkos::parallel_for(
        "2d_bc_x2p", range_p, Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i, j - nj, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i, j - nj, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i, j - nj, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i, j - nj, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i, j - nj, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i, j - nj, em::bx3);
        });
    } else {
      // non-periodic
      NTTError("2d boundary condition for minkowski not implemented");
    }
#elif (METRIC == SPHERICAL_METRIC) || (METRIC == QSPHERICAL_METRIC)
    // * * * * * * * * * * * * * * * *
    // axisymmetric spherical grid
    // * * * * * * * * * * * * * * * *
    // r = rmin boundary
    if (m_mblock.boundaries[0] == BoundaryCondition::USER) {
      m_pGen.userBCFields(t, m_sim_params, m_mblock);
    } else {
      NTTError("2d non-user boundary condition not implemented for curvilinear");
    }
    auto mblock {this->m_mblock};
    // theta = 0 boundary
    Kokkos::parallel_for(
      "2d_bc_theta0",
      NTTRange<Dim2>({0, 0}, {m_mblock.i1_max() + N_GHOSTS, m_mblock.i2_min() + 1}),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::bx2) = 0.0;
        mblock.em(i, j, em::ex3) = 0.0;
      });
    // theta = pi boundary
    Kokkos::parallel_for(
      "2d_bc_thetaPi",
      NTTRange<Dim2>({0, m_mblock.i2_max()},
                     {m_mblock.i1_max() + N_GHOSTS, m_mblock.i2_max() + N_GHOSTS}),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::bx2) = 0.0;
        mblock.em(i, j, em::ex3) = 0.0;
      });

    auto r_absorb {m_sim_params.metric_parameters()[2]};
    auto r_max {m_mblock.metric.x1_max};
    Kokkos::parallel_for("2d_absorbing bc",
                         m_mblock.rangeActiveCells(),
                         FieldBC_rmax<Dim2>(mblock, this->m_pGen, r_absorb, r_max));
#else
    (void)(index_t {});
    NTTError("2d boundary condition for metric not implemented");
#endif
  }

  /**
   * @brief 3d periodic field bc.
   *
   */
  template <>
  void PIC<Dim3>::fieldBoundaryConditions(const real_t&) {
    NTTError("not implemented");
  }

} // namespace ntt
