#include "global.h"
#include "pic.h"
#include "meshblock.h"

#if defined(SPHERICAL_METRIC) || defined(QSPHERICAL_METRIC)
#  include "pic_currents_bc_rmax.hpp"
#endif

namespace ntt {
  /**
   * @brief 1d current bc.
   *
   */
  template <>
  void PIC<Dim1>::currentBoundaryConditions(const real_t&) {

#ifdef MINKOWSKI_METRIC
    if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      // periodic
      auto mblock {this->m_mblock};
      auto ni {mblock.Ni1()};
      Kokkos::parallel_for(
        "1d_bc_x1m", mblock.rangeCells({CellLayer::minActiveLayer}), Lambda(index_t i) {
          mblock.cur(i, cur::jx1) += mblock.cur(i + ni, cur::jx1);
          mblock.cur(i, cur::jx2) += mblock.cur(i + ni, cur::jx2);
          mblock.cur(i, cur::jx3) += mblock.cur(i + ni, cur::jx3);
        });
      Kokkos::parallel_for(
        "1d_bc_x1p", mblock.rangeCells({CellLayer::maxActiveLayer}), Lambda(index_t i) {
          mblock.cur(i, cur::jx1) += mblock.cur(i - ni, cur::jx1);
          mblock.cur(i, cur::jx2) += mblock.cur(i - ni, cur::jx2);
          mblock.cur(i, cur::jx3) += mblock.cur(i - ni, cur::jx3);
        });
    } else {
      // non-periodic
      NTTError("2d boundary condition for minkowski not implemented");
    }
#else
    (void)(index_t {});
    NTTError("only minkowski possible in 1d");
#endif
  }

  /**
   * @brief 2d current bc.
   *
   */
  template <>
  void PIC<Dim2>::currentBoundaryConditions(const real_t&) {

#ifdef MINKOWSKI_METRIC
    /**
     * @note: no corners in each direction
     */
    if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      // periodic
      auto mblock {this->m_mblock};
      auto ni {mblock.Ni1()};
      Kokkos::parallel_for(
        "2d_bc_x1m",
        mblock.rangeCells({CellLayer::minActiveLayer, CellLayer::activeLayer}),
        Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i + ni, j, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i + ni, j, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i + ni, j, cur::jx3);
        });
      Kokkos::parallel_for(
        "2d_bc_x1p",
        mblock.rangeCells({CellLayer::maxActiveLayer, CellLayer::activeLayer}),
        Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i - ni, j, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i - ni, j, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i - ni, j, cur::jx3);
        });
    } else {
      // non-periodic
      NTTError("2d boundary condition for minkowski not implemented");
    }
    WaitAndSynchronize();
    if (m_mblock.boundaries[1] == BoundaryCondition::PERIODIC) {
      // periodic
      /**
       * @note: no corners
       */
      auto mblock {this->m_mblock};
      auto nj {mblock.Ni2()};
      Kokkos::parallel_for(
        "2d_bc_x2m",
        mblock.rangeCells({CellLayer::activeLayer, CellLayer::minActiveLayer}),
        Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i, j + nj, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i, j + nj, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i, j + nj, cur::jx3);
        });
      Kokkos::parallel_for(
        "2d_bc_x2p",
        mblock.rangeCells({CellLayer::activeLayer, CellLayer::maxActiveLayer}),
        Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i, j - nj, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i, j - nj, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i, j - nj, cur::jx3);
        });
    } else {
      // non-periodic
      NTTError("2d boundary condition for minkowski not implemented");
    }
    WaitAndSynchronize();
    /**
     * @note: corners treated separately
     */
    if ((m_mblock.boundaries[0] == BoundaryCondition::PERIODIC)
        && (m_mblock.boundaries[1] == BoundaryCondition::PERIODIC)) {
      auto mblock {this->m_mblock};
      auto ni {mblock.Ni1()};
      auto nj {mblock.Ni2()};
      Kokkos::parallel_for(
        "2d_bc_corner1",
        mblock.rangeCells({CellLayer::minActiveLayer, CellLayer::minActiveLayer}),
        Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i + ni, j + nj, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i + ni, j + nj, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i + ni, j + nj, cur::jx3);
        });
      Kokkos::parallel_for(
        "2d_bc_corner2",
        mblock.rangeCells({CellLayer::minActiveLayer, CellLayer::maxActiveLayer}),
        Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i + ni, j - nj, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i + ni, j - nj, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i + ni, j - nj, cur::jx3);
        });
      Kokkos::parallel_for(
        "2d_bc_corner3",
        mblock.rangeCells({CellLayer::maxActiveLayer, CellLayer::minActiveLayer}),
        Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i - ni, j + nj, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i - ni, j + nj, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i - ni, j + nj, cur::jx3);
        });
      Kokkos::parallel_for(
        "2d_bc_corner4",
        mblock.rangeCells({CellLayer::maxActiveLayer, CellLayer::maxActiveLayer}),
        Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i - ni, j - nj, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i - ni, j - nj, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i - ni, j - nj, cur::jx3);
        });
    }
#elif defined(SPHERICAL_METRIC) || defined(QSPHERICAL_METRIC)
    auto r_absorb {this->m_sim_params.metric_parameters()[2]};
    auto r_max {this->m_mblock.metric.x1_max};
    // !TODO: no need to do all cells
    Kokkos::parallel_for("2d_absorbing bc currs",
                         m_mblock.rangeActiveCells(),
                         CurrentBC_rmax<Dim2>(this->m_mblock, this->m_pGen, r_absorb, r_max));
#endif
  }

  /**
   * @brief 3d current bc.
   *
   */
  template <>
  void PIC<Dim3>::currentBoundaryConditions(const real_t&) {
    NTTError("not implemented");
  }

} // namespace ntt