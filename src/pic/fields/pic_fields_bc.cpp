#include "global.h"
#include "pic.h"

#if defined(SPHERICAL_METRIC) || defined(QSPHERICAL_METRIC)
#  include "pic_fields_bc_rmax.hpp"
#endif

namespace ntt {
  /**
   * @brief 1d periodic field bc.
   *
   */
  template <>
  void PIC<Dim1>::fieldBoundaryConditions(const real_t&) {

#ifdef MINKOWSKI_METRIC
    if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      auto mblock {this->m_mblock};
      auto ni {mblock.Ni1()};
      // in x1_min
      Kokkos::parallel_for(
        "1d_bc_x1m", mblock.rangeCells({CellLayer::minGhostLayer}), Lambda(index_t i) {
          mblock.em(i, em::ex1) = mblock.em(i + ni, em::ex1);
          mblock.em(i, em::ex2) = mblock.em(i + ni, em::ex2);
          mblock.em(i, em::ex3) = mblock.em(i + ni, em::ex3);
          mblock.em(i, em::bx1) = mblock.em(i + ni, em::bx1);
          mblock.em(i, em::bx2) = mblock.em(i + ni, em::bx2);
          mblock.em(i, em::bx3) = mblock.em(i + ni, em::bx3);
        });
      // in x1_max
      Kokkos::parallel_for(
        "1d_bc_x1p", mblock.rangeCells({CellLayer::maxGhostLayer}), Lambda(index_t i) {
          mblock.em(i, em::ex1) = mblock.em(i - ni, em::ex1);
          mblock.em(i, em::ex2) = mblock.em(i - ni, em::ex2);
          mblock.em(i, em::ex3) = mblock.em(i - ni, em::ex3);
          mblock.em(i, em::bx1) = mblock.em(i - ni, em::bx1);
          mblock.em(i, em::bx2) = mblock.em(i - ni, em::bx2);
          mblock.em(i, em::bx3) = mblock.em(i - ni, em::bx3);
        });
    } else {
      NTTHostError("boundary condition not implemented");
    }
#else
    (void)(index_t {});
    NTTHostError("only minkowski possible in 1d");
#endif
  }

  /**
   * @brief 2d periodic field bc.
   *
   */
  template <>
  void PIC<Dim2>::fieldBoundaryConditions(const real_t& t) {
#ifdef MINKOWSKI_METRIC
    (void)(t);
    if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      auto mblock {this->m_mblock};
      auto ni {mblock.Ni1()};
      Kokkos::parallel_for(
        "2d_bc_x1m",
        mblock.rangeCells({CellLayer::minGhostLayer, CellLayer::activeLayer}),
        Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i + ni, j, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i + ni, j, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i + ni, j, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i + ni, j, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i + ni, j, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i + ni, j, em::bx3);
        });
      Kokkos::parallel_for(
        "2d_bc_x1p",
        mblock.rangeCells({CellLayer::maxGhostLayer, CellLayer::activeLayer}),
        Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i - ni, j, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i - ni, j, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i - ni, j, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i - ni, j, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i - ni, j, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i - ni, j, em::bx3);
        });
    } else {
      // non-periodic
      NTTHostError("2d boundary condition for minkowski not implemented");
    }
    if (m_mblock.boundaries[1] == BoundaryCondition::PERIODIC) {
      auto mblock {this->m_mblock};
      auto nj {mblock.Ni2()};
      Kokkos::parallel_for(
        "2d_bc_x2m",
        mblock.rangeCells({CellLayer::activeLayer, CellLayer::minGhostLayer}),
        Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i, j + nj, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i, j + nj, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i, j + nj, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i, j + nj, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i, j + nj, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i, j + nj, em::bx3);
        });
      Kokkos::parallel_for(
        "2d_bc_x2p",
        mblock.rangeCells({CellLayer::activeLayer, CellLayer::maxGhostLayer}),
        Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i, j - nj, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i, j - nj, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i, j - nj, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i, j - nj, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i, j - nj, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i, j - nj, em::bx3);
        });
    } else {
      // non-periodic
      NTTHostError("2d boundary condition for minkowski not implemented");
    }

    if ((m_mblock.boundaries[1] == BoundaryCondition::PERIODIC)
        && (m_mblock.boundaries[1] == BoundaryCondition::PERIODIC)) {
      auto mblock {this->m_mblock};
      auto ni {mblock.Ni1()};
      auto nj {mblock.Ni2()};
      Kokkos::parallel_for(
        "2d_bc_corner1",
        mblock.rangeCells({CellLayer::minGhostLayer, CellLayer::minGhostLayer}),
        Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i + ni, j + nj, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i + ni, j + nj, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i + ni, j + nj, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i + ni, j + nj, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i + ni, j + nj, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i + ni, j + nj, em::bx3);
        });
      Kokkos::parallel_for(
        "2d_bc_corner2",
        mblock.rangeCells({CellLayer::minGhostLayer, CellLayer::maxGhostLayer}),
        Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i + ni, j - nj, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i + ni, j - nj, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i + ni, j - nj, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i + ni, j - nj, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i + ni, j - nj, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i + ni, j - nj, em::bx3);
        });
      Kokkos::parallel_for(
        "2d_bc_corner3",
        mblock.rangeCells({CellLayer::maxGhostLayer, CellLayer::minGhostLayer}),
        Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i - ni, j + nj, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i - ni, j + nj, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i - ni, j + nj, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i - ni, j + nj, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i - ni, j + nj, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i - ni, j + nj, em::bx3);
        });
      Kokkos::parallel_for(
        "2d_bc_corner4",
        mblock.rangeCells({CellLayer::maxGhostLayer, CellLayer::maxGhostLayer}),
        Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i - ni, j - nj, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i - ni, j - nj, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i - ni, j - nj, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i - ni, j - nj, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i - ni, j - nj, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i - ni, j - nj, em::bx3);
        });
    }

#elif defined(SPHERICAL_METRIC) || defined(QSPHERICAL_METRIC)

    /* ----------------------- axisymmetric spherical grid ---------------------- */
    // r = rmin boundary
    if (m_mblock.boundaries[0] == BoundaryCondition::USER) {
      m_pGen.userBCFields(t, m_sim_params, m_mblock);
    } else {
      NTTHostError("2d non-user boundary condition not implemented for curvilinear");
    }
    auto mblock {this->m_mblock};
    // theta = 0 boundary
    Kokkos::parallel_for(
      "2d_bc_theta0",
      CreateRangePolicy<Dim2>({0, 0}, {m_mblock.i1_max() + N_GHOSTS, m_mblock.i2_min() + 1}),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::bx2) = 0.0;
        mblock.em(i, j, em::ex3) = 0.0;
      });
    // theta = pi boundary
    Kokkos::parallel_for(
      "2d_bc_thetaPi",
      CreateRangePolicy<Dim2>({0, m_mblock.i2_max()},
                              {m_mblock.i1_max() + N_GHOSTS, m_mblock.i2_max() + N_GHOSTS}),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::bx2) = 0.0;
        mblock.em(i, j, em::ex3) = 0.0;
      });

    auto r_absorb {m_sim_params.metric_parameters()[2]};
    auto r_max {m_mblock.metric.x1_max};
    // !TODO: no need to do all cells
    Kokkos::parallel_for("2d_absorbing bc",
                         m_mblock.rangeActiveCells(),
                         FieldBC_rmax<Dim2>(mblock, this->m_pGen, r_absorb, r_max));
#else
    (void)(index_t {});
    NTTHostError("2d boundary condition for metric not implemented");
#endif
  }

  /**
   * @brief 3d periodic field bc.
   *
   */
  template <>
  void PIC<Dim3>::fieldBoundaryConditions(const real_t&) {
    NTTHostError("not implemented");
  }

} // namespace ntt
