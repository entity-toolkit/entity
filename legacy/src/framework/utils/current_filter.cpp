#include "wrapper.h"
// #include "current_filter.hpp"

namespace ntt {

  // template <>
  // void CurrentFilter<Dim1>::synchronizeGhostZones() const {
  //   auto ni {m_mesh.Ni1()};
  //   auto mesh {m_mesh};
  //   if (mesh.boundaries[0] == BoundaryCondition::PERIODIC) {
  //     Kokkos::parallel_for(
  //       "1d_gh_x1m", mesh.rangeCells({CellLayer::minGhostLayer}), Lambda(index_t i) {
  //         m_cur(i, cur::jx1) = m_cur(i + ni, cur::jx1);
  //         m_cur(i, cur::jx2) = m_cur(i + ni, cur::jx2);
  //         m_cur(i, cur::jx3) = m_cur(i + ni, cur::jx3);
  //       });
  //     Kokkos::parallel_for(
  //       "1d_gh_x1p", mesh.rangeCells({CellLayer::maxGhostLayer}), Lambda(index_t i) {
  //         m_cur(i, cur::jx1) = m_cur(i - ni, cur::jx1);
  //         m_cur(i, cur::jx2) = m_cur(i - ni, cur::jx2);
  //         m_cur(i, cur::jx3) = m_cur(i - ni, cur::jx3);
  //       });
  //   }
  // }

  // template <>
  // void CurrentFilter<Dim2>::synchronizeGhostZones() const {
  //   auto ni {m_mesh.Ni1()};
  //   auto nj {m_mesh.Ni2()};
  //   auto mesh {this->m_mesh};
  //   if (mesh.boundaries[0] == BoundaryCondition::PERIODIC) {
  //     Kokkos::parallel_for(
  //       "2d_gh_x1m",
  //       mesh.rangeCells({CellLayer::minGhostLayer, CellLayer::activeLayer}),
  //       Lambda(index_t i, index_t j) {
  //         m_cur(i, j, cur::jx1) = m_cur(i + ni, j, cur::jx1);
  //         m_cur(i, j, cur::jx2) = m_cur(i + ni, j, cur::jx2);
  //         m_cur(i, j, cur::jx3) = m_cur(i + ni, j, cur::jx3);
  //       });
  //     Kokkos::parallel_for(
  //       "2d_gh_x1p",
  //       mesh.rangeCells({CellLayer::maxGhostLayer, CellLayer::activeLayer}),
  //       Lambda(index_t i, index_t j) {
  //         m_cur(i, j, cur::jx1) = m_cur(i - ni, j, cur::jx1);
  //         m_cur(i, j, cur::jx2) = m_cur(i - ni, j, cur::jx2);
  //         m_cur(i, j, cur::jx3) = m_cur(i - ni, j, cur::jx3);
  //       });
  //   }
  //   if (mesh.boundaries[1] == BoundaryCondition::PERIODIC) {
  //     Kokkos::parallel_for(
  //       "2d_gh_x2m",
  //       mesh.rangeCells({CellLayer::activeLayer, CellLayer::minGhostLayer}),
  //       Lambda(index_t i, index_t j) {
  //         m_cur(i, j, cur::jx1) = m_cur(i, j + nj, cur::jx1);
  //         m_cur(i, j, cur::jx2) = m_cur(i, j + nj, cur::jx2);
  //         m_cur(i, j, cur::jx3) = m_cur(i, j + nj, cur::jx3);
  //       });
  //     Kokkos::parallel_for(
  //       "2d_gh_x2p",
  //       mesh.rangeCells({CellLayer::activeLayer, CellLayer::maxGhostLayer}),
  //       Lambda(index_t i, index_t j) {
  //         m_cur(i, j, cur::jx1) = m_cur(i, j - nj, cur::jx1);
  //         m_cur(i, j, cur::jx2) = m_cur(i, j - nj, cur::jx2);
  //         m_cur(i, j, cur::jx3) = m_cur(i, j - nj, cur::jx3);
  //       });
  //   }
  //   if ((mesh.boundaries[0] == BoundaryCondition::PERIODIC)
  //       && (mesh.boundaries[1] == BoundaryCondition::PERIODIC)) {
  //     Kokkos::parallel_for(
  //       "2d_bc_corner1",
  //       mesh.rangeCells({CellLayer::minGhostLayer, CellLayer::minGhostLayer}),
  //       Lambda(index_t i, index_t j) {
  //         m_cur(i, j, cur::jx1) = m_cur(i + ni, j + nj, cur::jx1);
  //         m_cur(i, j, cur::jx2) = m_cur(i + ni, j + nj, cur::jx2);
  //         m_cur(i, j, cur::jx3) = m_cur(i + ni, j + nj, cur::jx3);
  //       });
  //     Kokkos::parallel_for(
  //       "2d_bc_corner2",
  //       mesh.rangeCells({CellLayer::minGhostLayer, CellLayer::maxGhostLayer}),
  //       Lambda(index_t i, index_t j) {
  //         m_cur(i, j, cur::jx1) = m_cur(i + ni, j - nj, cur::jx1);
  //         m_cur(i, j, cur::jx2) = m_cur(i + ni, j - nj, cur::jx2);
  //         m_cur(i, j, cur::jx3) = m_cur(i + ni, j - nj, cur::jx3);
  //       });
  //     Kokkos::parallel_for(
  //       "2d_bc_corner3",
  //       mesh.rangeCells({CellLayer::maxGhostLayer, CellLayer::minGhostLayer}),
  //       Lambda(index_t i, index_t j) {
  //         m_cur(i, j, cur::jx1) = m_cur(i - ni, j + nj, cur::jx1);
  //         m_cur(i, j, cur::jx2) = m_cur(i - ni, j + nj, cur::jx2);
  //         m_cur(i, j, cur::jx3) = m_cur(i - ni, j + nj, cur::jx3);
  //       });
  //     Kokkos::parallel_for(
  //       "2d_bc_corner4",
  //       mesh.rangeCells({CellLayer::maxGhostLayer, CellLayer::maxGhostLayer}),
  //       Lambda(index_t i, index_t j) {
  //         m_cur(i, j, cur::jx1) = m_cur(i - ni, j - nj, cur::jx1);
  //         m_cur(i, j, cur::jx2) = m_cur(i - ni, j - nj, cur::jx2);
  //         m_cur(i, j, cur::jx3) = m_cur(i - ni, j - nj, cur::jx3);
  //       });
  //   }
  // }

  // template <>
  // void CurrentFilter<Dim3>::synchronizeGhostZones() const {}

} // namespace ntt

// template struct ntt::CurrentFilter<ntt::Dim1>;
// template struct ntt::CurrentFilter<ntt::Dim2>;
// template struct ntt::CurrentFilter<ntt::Dim3>;
