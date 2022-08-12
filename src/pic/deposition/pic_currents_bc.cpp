#include "global.h"
#include "pic.h"
#include "pic_currents_bc.hpp"
#include "meshblock.h"

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
  void PIC<Dim1>::currentBoundaryConditions(const real_t&) {

#if (METRIC == MINKOWSKI_METRIC)

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
  void PIC<Dim2>::currentBoundaryConditions(const real_t&) {

#if (METRIC == MINKOWSKI_METRIC)
    /**
     * @note: no corners in each direction
     */
    if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      // periodic
      auto ni {m_mblock.Ni1()};
      auto mblock {this->m_mblock};
      Kokkos::parallel_for(
        "2d_bc_x1m",
        mblock.rangeCells({CellLayer::minActiveLayer, CellLayer::allActiveLayer}),
        Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i + ni, j, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i + ni, j, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i + ni, j, cur::jx3);
        });
      Kokkos::parallel_for(
        "2d_bc_x1p",
        mblock.rangeCells({CellLayer::maxActiveLayer, CellLayer::allActiveLayer}),
        Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i - ni, j, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i - ni, j, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i - ni, j, cur::jx3);
        });
    } else {
      // non-periodic
      NTTError("2d boundary condition for minkowski not implemented");
    }
    NTTWait();
    if (m_mblock.boundaries[1] == BoundaryCondition::PERIODIC) {
      // periodic
      /**
       * @note: no corners
       */
      auto nj {m_mblock.Ni2()};
      auto mblock {this->m_mblock};
      auto range_m = NTTRange<Dim2>({mblock.i1_min(), mblock.i2_min()},
                                    {mblock.i1_max(), mblock.i2_min() + N_GHOSTS});
      auto range_p = NTTRange<Dim2>({mblock.i1_min(), mblock.i2_max() - N_GHOSTS},
                                    {mblock.i1_max(), mblock.i2_max()});
      Kokkos::parallel_for(
        "2d_bc_x1m",
        mblock.rangeCells({CellLayer::allActiveLayer, CellLayer::minActiveLayer}),
        Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i, j + nj, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i, j + nj, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i, j + nj, cur::jx3);
        });
      Kokkos::parallel_for(
        "2d_bc_x1p",
        mblock.rangeCells({CellLayer::allActiveLayer, CellLayer::maxActiveLayer}),
        Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i, j - nj, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i, j - nj, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i, j - nj, cur::jx3);
        });
    } else {
      // non-periodic
      NTTError("2d boundary condition for minkowski not implemented");
    }
    NTTWait();
    /**
     * @note: corners treated separately
     */
    if ((m_mblock.boundaries[1] == BoundaryCondition::PERIODIC)
        && (m_mblock.boundaries[1] == BoundaryCondition::PERIODIC)) {
      auto ni {m_mblock.Ni1()};
      auto nj {m_mblock.Ni2()};
      auto mblock {this->m_mblock};
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
#elif (METRIC == SPHERICAL_METRIC) || (METRIC == QSPHERICAL_METRIC)

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
  void PIC<Dim3>::currentBoundaryConditions(const real_t&) {
    NTTError("not implemented");
  }

} // namespace ntt
