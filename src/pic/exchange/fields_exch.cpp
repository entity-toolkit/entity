#include "wrapper.h"

#include "fields.h"
#include "meshblock.h"
#include "pic.h"

namespace ntt {
#ifdef MINKOWSKI_METRIC
  /**
   * @brief 1d periodic field bc (minkowski).
   */
  template <>
  void PIC<Dim1>::FieldsExchange() {
    auto& mblock = this->meshblock;
    if (mblock.boundaries[0][0] == BoundaryCondition::PERIODIC) {
      auto ni = mblock.Ni1();
      // in x1_min
      Kokkos::parallel_for(
        "1d_bc_x1m", mblock.rangeCells({ CellLayer::minGhostLayer }), Lambda(index_t i) {
          for (auto& comp : { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
            mblock.em(i, comp) = mblock.em(i + ni, comp);
          }
        });
      // in x1_max
      Kokkos::parallel_for(
        "1d_bc_x1p", mblock.rangeCells({ CellLayer::maxGhostLayer }), Lambda(index_t i) {
          for (auto& comp : { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
            mblock.em(i, comp) = mblock.em(i - ni, comp);
          }
        });
    } else {
      NTTHostError("boundary condition not implemented");
    }
  }

  /**
   * @brief 2d periodic field bc.
   *
   */
  template <>
  void PIC<Dim2>::FieldsExchange() {
    auto& mblock = this->meshblock;
    if (mblock.boundaries[0][0] == BoundaryCondition::PERIODIC) {
      auto ni = mblock.Ni1();
      Kokkos::parallel_for(
        "2d_bc_x1m",
        mblock.rangeCells({ CellLayer::minGhostLayer, CellLayer::activeLayer }),
        Lambda(index_t i, index_t j) {
          for (auto& comp : { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
            mblock.em(i, j, comp) = mblock.em(i + ni, j, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_bc_x1p",
        mblock.rangeCells({ CellLayer::maxGhostLayer, CellLayer::activeLayer }),
        Lambda(index_t i, index_t j) {
          for (auto& comp : { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
            mblock.em(i, j, comp) = mblock.em(i - ni, j, comp);
          }
        });
    } else {
      // non-periodic
      NTTHostError("2d boundary condition for minkowski not implemented");
    }
    if (mblock.boundaries[1][0] == BoundaryCondition::PERIODIC) {
      auto nj = mblock.Ni2();
      Kokkos::parallel_for(
        "2d_bc_x2m",
        mblock.rangeCells({ CellLayer::activeLayer, CellLayer::minGhostLayer }),
        Lambda(index_t i, index_t j) {
          for (auto& comp : { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
            mblock.em(i, j, comp) = mblock.em(i, j + nj, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_bc_x2p",
        mblock.rangeCells({ CellLayer::activeLayer, CellLayer::maxGhostLayer }),
        Lambda(index_t i, index_t j) {
          for (auto& comp : { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
            mblock.em(i, j, comp) = mblock.em(i, j - nj, comp);
          }
        });
    } else {
      // non-periodic
      NTTHostError("2d boundary condition for minkowski not implemented");
    }

    if ((mblock.boundaries[0][0] == BoundaryCondition::PERIODIC)
        && (mblock.boundaries[1][0] == BoundaryCondition::PERIODIC)) {
      auto ni = mblock.Ni1();
      auto nj = mblock.Ni2();
      Kokkos::parallel_for(
        "2d_bc_corner1",
        mblock.rangeCells({ CellLayer::minGhostLayer, CellLayer::minGhostLayer }),
        Lambda(index_t i, index_t j) {
          for (auto& comp : { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
            mblock.em(i, j, comp) = mblock.em(i + ni, j + nj, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_bc_corner2",
        mblock.rangeCells({ CellLayer::minGhostLayer, CellLayer::maxGhostLayer }),
        Lambda(index_t i, index_t j) {
          for (auto& comp : { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
            mblock.em(i, j, comp) = mblock.em(i + ni, j - nj, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_bc_corner3",
        mblock.rangeCells({ CellLayer::maxGhostLayer, CellLayer::minGhostLayer }),
        Lambda(index_t i, index_t j) {
          for (auto& comp : { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
            mblock.em(i, j, comp) = mblock.em(i - ni, j + nj, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_bc_corner4",
        mblock.rangeCells({ CellLayer::maxGhostLayer, CellLayer::maxGhostLayer }),
        Lambda(index_t i, index_t j) {
          for (auto& comp : { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
            mblock.em(i, j, comp) = mblock.em(i - ni, j - nj, comp);
          }
        });
    }
  }

  /**
   * @brief 3d periodic field bc.
   *
   */
  template <>
  void PIC<Dim3>::FieldsExchange() {
    NTTHostError("not implemented");
  }
#else
  template <Dimension D>
  void PIC<D>::FieldsExchange() {}
#endif

}    // namespace ntt

#ifndef MINKOWSKI_METRIC

template void ntt::PIC<ntt::Dim1>::FieldsExchange();
template void ntt::PIC<ntt::Dim2>::FieldsExchange();
template void ntt::PIC<ntt::Dim3>::FieldsExchange();

#endif