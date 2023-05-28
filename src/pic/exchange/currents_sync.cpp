#include "wrapper.h"

#include "io/output.h"
#include "meshblock/meshblock.h"
#include "pic.h"

namespace ntt {
#ifdef MINKOWSKI_METRIC
  template <>
  void PIC<Dim1>::CurrentsSynchronize() {
    auto& mblock = this->meshblock;
    NTTHostErrorIf(mblock.boundaries[0][0] != BoundaryCondition::PERIODIC,
                   "1d minkowski only supports periodic boundaries");
    const auto i1min = mblock.i1_min(), i1max = mblock.i1_max(), ni1 = mblock.Ni1();
    Kokkos::parallel_for(
      "CurrentsSynchronize", N_GHOSTS, Lambda(index_t i) {
#  pragma unroll
        for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
          mblock.cur(i1min + i, comp) += mblock.cur(i1min + i + ni1, comp);
          mblock.cur(i1max - N_GHOSTS + i, comp)
            += mblock.cur(i1max - N_GHOSTS + i - ni1, comp);
        }
      });
  }

  template <>
  void PIC<Dim2>::CurrentsSynchronize() {
    auto& mblock = this->meshblock;

    NTTHostErrorIf((mblock.boundaries[0][0] != BoundaryCondition::PERIODIC)
                     || (mblock.boundaries[1][0] != BoundaryCondition::PERIODIC),
                   "2d minkowski only supports periodic boundaries");

    const auto i1min = mblock.i1_min(), i1max = mblock.i1_max(), ni1 = mblock.Ni1();
    const auto i2min = mblock.i2_min(), i2max = mblock.i2_max(), ni2 = mblock.Ni2();

    Kokkos::parallel_for(
      "CurrentsSynchronize-1",
      CreateRangePolicy<Dim2>({ 0, i2min }, { N_GHOSTS, i2max }),
      Lambda(index_t i, index_t j) {
#  pragma unroll
        for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
          mblock.cur(i1min + i, j, comp) += mblock.cur(i1min + i + ni1, j, comp);
          mblock.cur(i1max - N_GHOSTS + i, j, comp)
            += mblock.cur(i1max - N_GHOSTS + i - ni1, j, comp);
        }
      });
    Kokkos::parallel_for(
      "CurrentsSynchronize-2",
      CreateRangePolicy<Dim2>({ i1min, 0 }, { i1max, N_GHOSTS }),
      Lambda(index_t i, index_t j) {
#  pragma unroll
        for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
          mblock.cur(i, i2min + j, comp) += mblock.cur(i, i2min + j + ni2, comp);
          mblock.cur(i, i2max - N_GHOSTS + j, comp)
            += mblock.cur(i, i2max - N_GHOSTS + j - ni2, comp);
        }
      });
    // corners
    Kokkos::parallel_for(
      "CurrentsSynchronize-3",
      CreateRangePolicy<Dim2>({ 0, 0 }, { N_GHOSTS, N_GHOSTS }),
      Lambda(index_t i, index_t j) {
#  pragma unroll
        for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
          mblock.cur(i1min + i, i2min + j, comp)
            += mblock.cur(i1min + i + ni1, i2min + j + ni2, comp);
          mblock.cur(i1max - N_GHOSTS + i, i2min + j, comp)
            += mblock.cur(i1max - N_GHOSTS + i - ni1, i2min + j + ni2, comp);
          mblock.cur(i1min + i, i2max - N_GHOSTS + j, comp)
            += mblock.cur(i1min + i + ni1, i2max - N_GHOSTS + j - ni2, comp);
          mblock.cur(i1max - N_GHOSTS + i, i2max - N_GHOSTS + j, comp)
            += mblock.cur(i1max - N_GHOSTS + i - ni1, i2max - N_GHOSTS + j - ni2, comp);
        }
      });
  }

  template <>
  void PIC<Dim3>::CurrentsSynchronize() {
    auto& mblock = this->meshblock;

    NTTHostErrorIf((mblock.boundaries[0][0] != BoundaryCondition::PERIODIC)
                     || (mblock.boundaries[1][0] != BoundaryCondition::PERIODIC)
                     || (mblock.boundaries[2][0] != BoundaryCondition::PERIODIC),
                   "3d minkowski only supports periodic boundaries");

    const auto i1min = mblock.i1_min(), i1max = mblock.i1_max(), ni1 = mblock.Ni1();
    const auto i2min = mblock.i2_min(), i2max = mblock.i2_max(), ni2 = mblock.Ni2();
    const auto i3min = mblock.i3_min(), i3max = mblock.i3_max(), ni3 = mblock.Ni3();

    Kokkos::parallel_for(
      "CurrentsSynchronize-1",
      CreateRangePolicy<Dim3>({ 0, i2min, i3min }, { N_GHOSTS, i2max, i3max }),
      Lambda(index_t i, index_t j, index_t k) {
#  pragma unroll
        for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
          mblock.cur(i1min + i, j, k, comp) += mblock.cur(i1min + i + ni1, j, k, comp);
          mblock.cur(i1max - N_GHOSTS + i, j, k, comp)
            += mblock.cur(i1max - N_GHOSTS + i - ni1, j, k, comp);
        }
      });
    Kokkos::parallel_for(
      "CurrentsSynchronize-2",
      CreateRangePolicy<Dim3>({ i1min, 0, i3min }, { i1max, N_GHOSTS, i3max }),
      Lambda(index_t i, index_t j, index_t k) {
#  pragma unroll
        for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
          mblock.cur(i, i2min + j, k, comp) += mblock.cur(i, i2min + j + ni2, k, comp);
          mblock.cur(i, i2max - N_GHOSTS + j, k, comp)
            += mblock.cur(i, i2max - N_GHOSTS + j - ni2, k, comp);
        }
      });
    Kokkos::parallel_for(
      "CurrentsSynchronize-3",
      CreateRangePolicy<Dim3>({ i1min, i2min, 0 }, { i1max, i2max, N_GHOSTS }),
      Lambda(index_t i, index_t j, index_t k) {
#  pragma unroll
        for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
          mblock.cur(i, j, i3min + k, comp) += mblock.cur(i, j, i3min + k + ni3, comp);
          mblock.cur(i, j, i3max - N_GHOSTS + k, comp)
            += mblock.cur(i, j, i2max - N_GHOSTS + k - ni3, comp);
        }
      });

    // extended corners
    Kokkos::parallel_for(
      "CurrentsSynchronize-4",
      CreateRangePolicy<Dim3>({ 0, 0, i3min }, { N_GHOSTS, N_GHOSTS, i3max }),
      Lambda(index_t i, index_t j, index_t k) {
#  pragma unroll
        for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
          mblock.cur(i1min + i, i2min + j, k, comp)
            += mblock.cur(i1min + i + ni1, i2min + j + ni2, k, comp);
          mblock.cur(i1max - N_GHOSTS + i, i2min + j, k, comp)
            += mblock.cur(i1max - N_GHOSTS + i - ni1, i2min + j + ni2, k, comp);
          mblock.cur(i1min + i, i2max - N_GHOSTS + j, k, comp)
            += mblock.cur(i1min + i + ni1, i2max - N_GHOSTS + j - ni2, k, comp);
          mblock.cur(i1max - N_GHOSTS + i, i2max - N_GHOSTS + j, k, comp)
            += mblock.cur(i1max - N_GHOSTS + i - ni1, i2max - N_GHOSTS + j - ni2, k, comp);
        }
      });
    Kokkos::parallel_for(
      "CurrentsSynchronize-5",
      CreateRangePolicy<Dim3>({ i1min, 0, 0 }, { i1max, N_GHOSTS, N_GHOSTS }),
      Lambda(index_t i, index_t j, index_t k) {
#  pragma unroll
        for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
          mblock.cur(i, i2min + j, i3min + k, comp)
            += mblock.cur(i, i2min + j + ni2, i3min + k - ni3, comp);
          mblock.cur(i, i2max - N_GHOSTS + j, i3min + k, comp)
            += mblock.cur(i, i2max - N_GHOSTS + j - ni2, i3min + k - ni3, comp);
          mblock.cur(i, i2min + j, i3max - N_GHOSTS + k, comp)
            += mblock.cur(i, i2min + j + ni2, i3max - N_GHOSTS + k - ni2, comp);
          mblock.cur(i, i2max - N_GHOSTS + j, i3max - N_GHOSTS + k, comp)
            += mblock.cur(i, i2max - N_GHOSTS + j - ni2, i3max - N_GHOSTS + k - ni2, comp);
        }
      });
    Kokkos::parallel_for(
      "CurrentsSynchronize-6",
      CreateRangePolicy<Dim3>({ 0, i2min, 0 }, { N_GHOSTS, i2max, N_GHOSTS }),
      Lambda(index_t i, index_t j, index_t k) {
#  pragma unroll
        for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
          mblock.cur(i1min + i, j, i3min + k, comp)
            += mblock.cur(i1min + i + ni1, j, i3min + k - ni3, comp);
          mblock.cur(i1max - N_GHOSTS + i, j, i3min + k, comp)
            += mblock.cur(i1max - N_GHOSTS + i - ni1, j, i3min + k - ni3, comp);
          mblock.cur(i1min + i, j, i3max - N_GHOSTS + k, comp)
            += mblock.cur(i1min + i + ni1, j, i3max - N_GHOSTS + k - ni3, comp);
          mblock.cur(i1max - N_GHOSTS + i, j, i3max - N_GHOSTS + k, comp)
            += mblock.cur(i1max - N_GHOSTS + i - ni1, j, i3max - N_GHOSTS + k - ni3, comp);
        }
      });

    // corners
    Kokkos::parallel_for(
      "CurrentsSynchronize-7",
      CreateRangePolicy<Dim3>({ 0, 0, 0 }, { N_GHOSTS, N_GHOSTS, N_GHOSTS }),
      Lambda(index_t i, index_t j, index_t k) {
#  pragma unroll
        for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
          mblock.cur(i1min + i, i2min + j, i3min + k, comp)
            += mblock.cur(i1min + i + ni1, i2min + j + ni2, i3min + k - ni3, comp);
          mblock.cur(i1max - N_GHOSTS + i, i2min + j, i3min + k, comp)
            += mblock.cur(i1max - N_GHOSTS + i - ni1, i2min + j + ni2, i3min + k - ni3, comp);
          mblock.cur(i1min + i, i2max - N_GHOSTS + j, i3min + k, comp)
            += mblock.cur(i1min + i + ni1, i2max - N_GHOSTS + j - ni2, i3min + k - ni3, comp);
          mblock.cur(i1max - N_GHOSTS + i, i2max - N_GHOSTS + j, i3min + k, comp)
            += mblock.cur(
              i1max - N_GHOSTS + i - ni1, i2max - N_GHOSTS + j - ni2, i3min + k - ni3, comp);
          mblock.cur(i1min + i, i2min + j, i3max - N_GHOSTS + k, comp)
            += mblock.cur(i1min + i + ni1, i2min + j + ni2, i3max - N_GHOSTS + k - ni3, comp);
          mblock.cur(i1max - N_GHOSTS + i, i2min + j, i3max - N_GHOSTS + k, comp)
            += mblock.cur(
              i1max - N_GHOSTS + i - ni1, i2min + j + ni2, i3max - N_GHOSTS + k - ni3, comp);
          mblock.cur(i1min + i, i2max - N_GHOSTS + j, i3max - N_GHOSTS + k, comp)
            += mblock.cur(
              i1min + i + ni1, i2max - N_GHOSTS + j - ni2, i3max - N_GHOSTS + k - ni3, comp);
          mblock.cur(i1max - N_GHOSTS + i, i2max - N_GHOSTS + j, i3max - N_GHOSTS + k, comp)
            += mblock.cur(i1max - N_GHOSTS + i - ni1,
                          i2max - N_GHOSTS + j - ni2,
                          i3max - N_GHOSTS + k - ni3,
                          comp);
        }
      });
  }

#else
  template <Dimension D>
  void PIC<D>::CurrentsSynchronize() {}

#endif

}    // namespace ntt

#ifndef MINKOWSKI_METRIC
template void ntt::PIC<ntt::Dim1>::CurrentsSynchronize();
template void ntt::PIC<ntt::Dim2>::CurrentsSynchronize();
template void ntt::PIC<ntt::Dim3>::CurrentsSynchronize();
#endif

// Kokkos::parallel_for(
//   "1d_bc_x1p", mblock.rangeCells({ CellLayer::maxActiveLayer }), Lambda(index_t i) {
//     mblock.cur(i, cur::jx1) += mblock.cur(i - ni1, cur::jx1);
//     mblock.cur(i, cur::jx2) += mblock.cur(i - ni1, cur::jx2);
//     mblock.cur(i, cur::jx3) += mblock.cur(i - ni1, cur::jx3);
//   });

//     Kokkos::parallel_for(
//       "2d_bc_x1p",
//       mblock.rangeCells({ CellLayer::maxActiveLayer, CellLayer::activeLayer }),
//       Lambda(index_t i, index_t j) {
// #  pragma unroll
//         for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
//           mblock.cur(i, j, comp) += mblock.cur(i - ni1, j, comp);
//         }
//       });

//     Kokkos::parallel_for(
//       "2d_bc_x2m",
//       mblock.rangeCells({ CellLayer::activeLayer, CellLayer::minActiveLayer }),
//       Lambda(index_t i, index_t j) {
// #  pragma unroll
//         for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
//           mblock.cur(i, j, comp) += mblock.cur(i, j + ni2, comp);
//         }
//       });
//     Kokkos::parallel_for(
//       "2d_bc_x2p",
//       mblock.rangeCells({ CellLayer::activeLayer, CellLayer::maxActiveLayer }),
//       Lambda(index_t i, index_t j) {
// #  pragma unroll
//         for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
//           mblock.cur(i, j, comp) += mblock.cur(i, j - ni2, comp);
//         }
//       });

// corners treated separately
//     Kokkos::parallel_for(
//       "2d_bc_corner1",
//       mblock.rangeCells({ CellLayer::minActiveLayer, CellLayer::minActiveLayer }),
//       Lambda(index_t i, index_t j) {
// #  pragma unroll
//         for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
//           mblock.cur(i, j, comp) += mblock.cur(i + ni1, j + ni2, comp);
//         }
//       });
//     Kokkos::parallel_for(
//       "2d_bc_corner2",
//       mblock.rangeCells({ CellLayer::minActiveLayer, CellLayer::maxActiveLayer }),
//       Lambda(index_t i, index_t j) {
// #  pragma unroll
//         for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
//           mblock.cur(i, j, comp) += mblock.cur(i + ni1, j - ni2, comp);
//         }
//       });
//     Kokkos::parallel_for(
//       "2d_bc_corner3",
//       mblock.rangeCells({ CellLayer::maxActiveLayer, CellLayer::minActiveLayer }),
//       Lambda(index_t i, index_t j) {
// #  pragma unroll
//         for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
//           mblock.cur(i, j, comp) += mblock.cur(i - ni1, j + ni2, comp);
//         }
//       });
//     Kokkos::parallel_for(
//       "2d_bc_corner4",
//       mblock.rangeCells({ CellLayer::maxActiveLayer, CellLayer::maxActiveLayer }),
//       Lambda(index_t i, index_t j) {
// #  pragma unroll
//         for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
//           mblock.cur(i, j, comp) += mblock.cur(i - ni1, j - ni2, comp);
//         }
//       });