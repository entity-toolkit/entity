#include "wrapper.h"
#include "meshblock.h"
#include "fields.h"
#include "pic.h"

namespace ntt {
#ifdef MINKOWSKI_METRIC
  template <>
  void PIC<Dim1>::CurrentsSynchronize() {
    auto& mblock = this->meshblock;
    if (mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      // periodic

      auto ni = mblock.Ni1();
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
      NTTHostError("2d boundary condition for minkowski not implemented");
    }
  }

  template <>
  void PIC<Dim2>::CurrentsSynchronize() {
    /**
     * @note: no corners in each direction
     */
    auto& mblock = this->meshblock;
    if (mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      auto ni = mblock.Ni1();
      Kokkos::parallel_for(
        "2d_bc_x1m",
        mblock.rangeCells({CellLayer::minActiveLayer, CellLayer::activeLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) += mblock.cur(i + ni, j, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_bc_x1p",
        mblock.rangeCells({CellLayer::maxActiveLayer, CellLayer::activeLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) += mblock.cur(i - ni, j, comp);
          }
        });
    } else {
      // non-periodic
      NTTHostError("2d boundary condition for minkowski not implemented");
    }
    WaitAndSynchronize();
    if (mblock.boundaries[1] == BoundaryCondition::PERIODIC) {
      /**
       * @note: no corners
       */
      auto nj = mblock.Ni2();
      Kokkos::parallel_for(
        "2d_bc_x2m",
        mblock.rangeCells({CellLayer::activeLayer, CellLayer::minActiveLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) += mblock.cur(i, j + nj, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_bc_x2p",
        mblock.rangeCells({CellLayer::activeLayer, CellLayer::maxActiveLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) += mblock.cur(i, j - nj, comp);
          }
        });
    } else {
      // non-periodic
      NTTHostError("2d boundary condition for minkowski not implemented");
    }
    WaitAndSynchronize();
    /**
     * @note: corners treated separately
     */
    if ((mblock.boundaries[0] == BoundaryCondition::PERIODIC)
        && (mblock.boundaries[1] == BoundaryCondition::PERIODIC)) {
      auto ni = mblock.Ni1();
      auto nj = mblock.Ni2();
      Kokkos::parallel_for(
        "2d_bc_corner1",
        mblock.rangeCells({CellLayer::minActiveLayer, CellLayer::minActiveLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) += mblock.cur(i + ni, j + nj, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_bc_corner2",
        mblock.rangeCells({CellLayer::minActiveLayer, CellLayer::maxActiveLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) += mblock.cur(i + ni, j - nj, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_bc_corner3",
        mblock.rangeCells({CellLayer::maxActiveLayer, CellLayer::minActiveLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) += mblock.cur(i - ni, j + nj, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_bc_corner4",
        mblock.rangeCells({CellLayer::maxActiveLayer, CellLayer::maxActiveLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) += mblock.cur(i - ni, j - nj, comp);
          }
        });
    }
  }

  template <>
  void PIC<Dim3>::CurrentsSynchronize() {
    NTTHostError("not implemented");
  }

#else
  template <Dimension D>
  void PIC<D>::CurrentsSynchronize() {}
#endif

  template <>
  void PIC<Dim1>::CurrentsExchange() {
    auto& mblock = this->meshblock;
    auto  ni     = mblock.Ni1();
    if (mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      Kokkos::parallel_for(
        "1d_gh_x1m", mblock.rangeCells({CellLayer::minGhostLayer}), Lambda(index_t i) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, comp) = mblock.cur(i + ni, comp);
          }
        });
      Kokkos::parallel_for(
        "1d_gh_x1p", mblock.rangeCells({CellLayer::maxGhostLayer}), Lambda(index_t i) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, comp) = mblock.cur(i - ni, comp);
          }
        });
    }
  }

  template <>
  void PIC<Dim2>::CurrentsExchange() {
    auto& mblock = this->meshblock;
    auto  ni     = mblock.Ni1();
    auto  nj     = mblock.Ni2();
    if (mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      Kokkos::parallel_for(
        "2d_gh_x1m",
        mblock.rangeCells({CellLayer::minGhostLayer, CellLayer::activeLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) = mblock.cur(i + ni, j, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_gh_x1p",
        mblock.rangeCells({CellLayer::maxGhostLayer, CellLayer::activeLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) = mblock.cur(i - ni, j, comp);
          }
        });
    }
    if (mblock.boundaries[1] == BoundaryCondition::PERIODIC) {
      Kokkos::parallel_for(
        "2d_gh_x2m",
        mblock.rangeCells({CellLayer::activeLayer, CellLayer::minGhostLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) = mblock.cur(i, j + nj, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_gh_x2p",
        mblock.rangeCells({CellLayer::activeLayer, CellLayer::maxGhostLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) = mblock.cur(i, j - nj, comp);
          }
        });
    }
    if ((mblock.boundaries[0] == BoundaryCondition::PERIODIC)
        && (mblock.boundaries[1] == BoundaryCondition::PERIODIC)) {
      Kokkos::parallel_for(
        "2d_bc_corner1",
        mblock.rangeCells({CellLayer::minGhostLayer, CellLayer::minGhostLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) = mblock.cur(i + ni, j + nj, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_bc_corner2",
        mblock.rangeCells({CellLayer::minGhostLayer, CellLayer::maxGhostLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) = mblock.cur(i + ni, j - nj, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_bc_corner3",
        mblock.rangeCells({CellLayer::maxGhostLayer, CellLayer::minGhostLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) = mblock.cur(i - ni, j + nj, comp);
          }
        });
      Kokkos::parallel_for(
        "2d_bc_corner4",
        mblock.rangeCells({CellLayer::maxGhostLayer, CellLayer::maxGhostLayer}),
        Lambda(index_t i, index_t j) {
          for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
            mblock.cur(i, j, comp) = mblock.cur(i - ni, j - nj, comp);
          }
        });
    }
  }

  template <>
  void PIC<Dim3>::CurrentsExchange() {}

} // namespace ntt


template void ntt::PIC<ntt::Dim1>::CurrentsSynchronize();
template void ntt::PIC<ntt::Dim2>::CurrentsSynchronize();
template void ntt::PIC<ntt::Dim3>::CurrentsSynchronize();

template void ntt::PIC<ntt::Dim1>::CurrentsExchange();
template void ntt::PIC<ntt::Dim2>::CurrentsExchange();
template void ntt::PIC<ntt::Dim3>::CurrentsExchange();