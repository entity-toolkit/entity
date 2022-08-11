#include "global.h"
#include "pic.h"
#include "pic_currents_bc.hpp"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {
  /**
   * @brief 1d periodic field bc.
   *
   */
  template <>
  void PIC<Dimension::ONE_D>::currentBoundaryConditions(const real_t&) {

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
  void PIC<Dimension::TWO_D>::currentBoundaryConditions(const real_t&) {

#if (METRIC == MINKOWSKI_METRIC)
    /**
     * @note: no corners in each direction
     */
    if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      // periodic
      auto ni {m_mblock.Ni()};
      auto mblock {this->m_mblock};
      auto range_m = NTTRange<Dimension::TWO_D>({mblock.i_min(), mblock.j_min()},
                                                {mblock.i_min() + N_GHOSTS, mblock.j_max()});
      auto range_p = NTTRange<Dimension::TWO_D>({mblock.i_max() - N_GHOSTS, mblock.j_min()},
                                                {mblock.i_max(), mblock.j_max()});
      Kokkos::parallel_for(
        "2d_bc_x1m", range_m, Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i + ni, j, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i + ni, j, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i + ni, j, cur::jx3);
        });
      Kokkos::parallel_for(
        "2d_bc_x1p", range_p, Lambda(index_t i, index_t j) {
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
      auto nj {m_mblock.Nj()};
      auto mblock {this->m_mblock};
      auto range_m = NTTRange<Dimension::TWO_D>({mblock.i_min(), mblock.j_min()},
                                                {mblock.i_max(), mblock.j_min() + N_GHOSTS});
      auto range_p = NTTRange<Dimension::TWO_D>({mblock.i_min(), mblock.j_max() - N_GHOSTS},
                                                {mblock.i_max(), mblock.j_max()});
      Kokkos::parallel_for(
        "2d_bc_x1m", range_m, Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i, j + nj, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i, j + nj, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i, j + nj, cur::jx3);
        });
      Kokkos::parallel_for(
        "2d_bc_x1p", range_p, Lambda(index_t i, index_t j) {
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
      auto ni {m_mblock.Ni()};
      auto nj {m_mblock.Nj()};
      auto mblock {this->m_mblock};
      auto range_corner1
        = NTTRange<Dimension::TWO_D>({mblock.i_min(), mblock.j_min()},
                                     {mblock.i_min() + N_GHOSTS, mblock.j_min() + N_GHOSTS});
      Kokkos::parallel_for(
        "2d_bc_corner1", range_corner1, Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i + ni, j + nj, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i + ni, j + nj, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i + ni, j + nj, cur::jx3);
        });
      auto range_corner2
        = NTTRange<Dimension::TWO_D>({mblock.i_min(), mblock.j_max() - N_GHOSTS},
                                     {mblock.i_min() + N_GHOSTS, mblock.j_max()});
      Kokkos::parallel_for(
        "2d_bc_corner2", range_corner2, Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i + ni, j - nj, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i + ni, j - nj, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i + ni, j - nj, cur::jx3);
        });
      auto range_corner3
        = NTTRange<Dimension::TWO_D>({mblock.i_max() - N_GHOSTS, mblock.j_min()},
                                     {mblock.i_max(), mblock.j_min() + N_GHOSTS});
      Kokkos::parallel_for(
        "2d_bc_corner3", range_corner3, Lambda(index_t i, index_t j) {
          mblock.cur(i, j, cur::jx1) += mblock.cur(i - ni, j + nj, cur::jx1);
          mblock.cur(i, j, cur::jx2) += mblock.cur(i - ni, j + nj, cur::jx2);
          mblock.cur(i, j, cur::jx3) += mblock.cur(i - ni, j + nj, cur::jx3);
        });
      auto range_corner4
        = NTTRange<Dimension::TWO_D>({mblock.i_max() - N_GHOSTS, mblock.j_max() - N_GHOSTS},
                                     {mblock.i_max(), mblock.j_max()});
      Kokkos::parallel_for(
        "2d_bc_corner4", range_corner4, Lambda(index_t i, index_t j) {
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
  void PIC<Dimension::THREE_D>::currentBoundaryConditions(const real_t&) {
    NTTError("not implemented");
  }

} // namespace ntt
