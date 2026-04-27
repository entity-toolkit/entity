#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/sorting.h"

#include "framework/containers/particles.h"
#include "framework/domain/grid.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <string>
#include <utility>
#include <vector>

namespace ntt {

  template <Dimension D, Coord::type C>
  auto Particles<D, C>::NpartsPerTagAndOffsets() const
    -> std::pair<std::vector<npart_t>, array_t<npart_t*>> {
    auto              this_tag = tag;
    const auto        num_tags = ntags();
    array_t<npart_t*> npptag { "nparts_per_tag", ntags() };

    // count # of particles per each tag
    auto npptag_scat = Kokkos::Experimental::create_scatter_view(npptag);
    Kokkos::parallel_for(
      "NpartPerTag",
      rangeActiveParticles(),
      Lambda(index_t p) {
        auto npptag_acc = npptag_scat.access();
        if (this_tag(p) < 0 || this_tag(p) >= static_cast<short>(num_tags)) {
          raise::KernelError(HERE, "Invalid tag value");
        }
        npptag_acc(this_tag(p)) += 1;
      });
    Kokkos::Experimental::contribute(npptag, npptag_scat);

    // copy the count to a vector on the host
    auto npptag_h = Kokkos::create_mirror_view(npptag);
    Kokkos::deep_copy(npptag_h, npptag);
    std::vector<npart_t> npptag_vec(num_tags);
    for (auto t { 0u }; t < num_tags; ++t) {
      npptag_vec[t] = npptag_h(t);
    }

    // count the offsets on the host and copy to device
    const array_t<npart_t*> tag_offsets("tag_offsets", num_tags - 3);
    auto tag_offsets_h = Kokkos::create_mirror_view(tag_offsets);

    tag_offsets_h(0) = npptag_vec[2]; // offset for tag = 3
    for (auto t { 1u }; t < num_tags - 3; ++t) {
      tag_offsets_h(t) = npptag_vec[t + 2] + tag_offsets_h(t - 1);
    }
    Kokkos::deep_copy(tag_offsets, tag_offsets_h);

    return { npptag_vec, tag_offsets };
  }

  template <typename T>
  void RemoveDeadInArray(array_t<T*>& arr, const array_t<npart_t*>& indices_alive) {
    const npart_t n_alive = indices_alive.extent(0);
    auto          buffer  = Kokkos::View<T*>("buffer", n_alive);
    Kokkos::parallel_for(
      "PopulateBufferAlive",
      n_alive,
      Lambda(index_t p) { buffer(p) = arr(indices_alive(p)); });

    Kokkos::deep_copy(
      Kokkos::subview(arr, std::make_pair(static_cast<npart_t>(0), n_alive)),
      buffer);
  }

  template <typename T>
  void RemoveDeadInArray(array_t<T**>& arr, const array_t<npart_t*>& indices_alive) {
    const npart_t n_alive = indices_alive.extent(0);
    auto          buffer  = array_t<T**> { "buffer", n_alive, arr.extent(1) };
    Kokkos::parallel_for(
      "PopulateBufferAlive",
      CreateRangePolicy<Dim::_2D>({ 0, 0 }, { n_alive, arr.extent(1) }),
      Lambda(index_t p, index_t l) { buffer(p, l) = arr(indices_alive(p), l); });

    Kokkos::deep_copy(
      Kokkos::subview(arr,
                      std::make_pair(static_cast<npart_t>(0), n_alive),
                      Kokkos::ALL),
      buffer);
  }

  template <Dimension D, Coord::type C>
  void Particles<D, C>::RemoveDead() {
    npart_t n_alive = 0, n_dead = 0;
    auto&   this_tag = tag;

    Kokkos::parallel_reduce(
      "CountDeadAlive",
      rangeActiveParticles(),
      Lambda(index_t p, npart_t & nalive, npart_t & ndead) {
        nalive += (this_tag(p) == ParticleTag::alive);
        ndead  += (this_tag(p) == ParticleTag::dead);
        if (this_tag(p) != ParticleTag::alive and this_tag(p) != ParticleTag::dead) {
          raise::KernelError(HERE, "wrong particle tag");
        }
      },
      n_alive,
      n_dead);

    const array_t<npart_t*> indices_alive { "indices_alive", n_alive };
    const array_t<npart_t*> alive_counter { "counter_alive", 1 };

    Kokkos::parallel_for(
      "AliveIndices",
      rangeActiveParticles(),
      Lambda(index_t p) {
        if (this_tag(p) == ParticleTag::alive) {
          const auto idx     = Kokkos::atomic_fetch_add(&alive_counter(0), 1);
          indices_alive(idx) = p;
        }
      });

    {
      auto alive_counter_h = Kokkos::create_mirror_view(alive_counter);
      Kokkos::deep_copy(alive_counter_h, alive_counter);
      raise::ErrorIf(alive_counter_h(0) != n_alive,
                     "error in finding alive particle indices",
                     HERE);
    }

    if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
      RemoveDeadInArray(i1, indices_alive);
      RemoveDeadInArray(i1_prev, indices_alive);
      RemoveDeadInArray(dx1, indices_alive);
      RemoveDeadInArray(dx1_prev, indices_alive);
    }

    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      RemoveDeadInArray(i2, indices_alive);
      RemoveDeadInArray(i2_prev, indices_alive);
      RemoveDeadInArray(dx2, indices_alive);
      RemoveDeadInArray(dx2_prev, indices_alive);
    }

    if constexpr (D == Dim::_3D) {
      RemoveDeadInArray(i3, indices_alive);
      RemoveDeadInArray(i3_prev, indices_alive);
      RemoveDeadInArray(dx3, indices_alive);
      RemoveDeadInArray(dx3_prev, indices_alive);
    }

    RemoveDeadInArray(ux1, indices_alive);
    RemoveDeadInArray(ux2, indices_alive);
    RemoveDeadInArray(ux3, indices_alive);
    RemoveDeadInArray(weight, indices_alive);

    if constexpr (D == Dim::_2D && C != Coord::Cartesian) {
      RemoveDeadInArray(phi, indices_alive);
    }

    if (npld_r() > 0) {
      RemoveDeadInArray(pld_r, indices_alive);
    }

    if (npld_i() > 0) {
      RemoveDeadInArray(pld_i, indices_alive);
    }

    Kokkos::Experimental::fill(
      "TagAliveParticles",
      Kokkos::DefaultExecutionSpace(),
      Kokkos::subview(this_tag, std::make_pair(static_cast<npart_t>(0), n_alive)),
      ParticleTag::alive);

    Kokkos::Experimental::fill(
      "TagDeadParticles",
      Kokkos::DefaultExecutionSpace(),
      Kokkos::subview(this_tag, std::make_pair(n_alive, n_alive + n_dead)),
      ParticleTag::dead);

    set_npart(n_alive);
    m_is_sorted = true;
  }

  template <Dimension D, Coord::type C>
  void Particles<D, C>::SortSpatially(const Grid<D>& grid) {
    const auto nx2         = grid.n_active(in::x2);
    const auto nx3         = grid.n_active(in::x3);
    const auto total_cells = grid.num_active();

    array_t<ncells_t*> cell_indices { "cell_indices", npart() };

    Kokkos::parallel_for(
      "FillCellIndices",
      rangeActiveParticles(),
      sort::PositionToCellIndex<D> { i1, i2, i3, tag, cell_indices, nx2, nx3, total_cells });
    const auto slice = range_tuple_t(0, npart());

    using sorter_op_t = Kokkos::BinOp1D<decltype(cell_indices)>;
    using sorter_t    = Kokkos::BinSort<decltype(cell_indices), sorter_op_t>;
    auto bin_op       = sorter_op_t { static_cast<int>(total_cells + 1u),
                                0u,
                                total_cells + 1u };
    auto sorter       = sorter_t { cell_indices, bin_op, false };
    sorter.create_permute_vector();
    if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
      sorter.sort(Kokkos::subview(i1, slice));
      sorter.sort(Kokkos::subview(i1_prev, slice));
      sorter.sort(Kokkos::subview(dx1, slice));
      sorter.sort(Kokkos::subview(dx1_prev, slice));
    }
    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      sorter.sort(Kokkos::subview(i2, slice));
      sorter.sort(Kokkos::subview(i2_prev, slice));
      sorter.sort(Kokkos::subview(dx2, slice));
      sorter.sort(Kokkos::subview(dx2_prev, slice));
    }
    if constexpr (D == Dim::_3D) {
      sorter.sort(Kokkos::subview(i3, slice));
      sorter.sort(Kokkos::subview(i3_prev, slice));
      sorter.sort(Kokkos::subview(dx3, slice));
      sorter.sort(Kokkos::subview(dx3_prev, slice));
    }
    sorter.sort(Kokkos::subview(ux1, slice));
    sorter.sort(Kokkos::subview(ux2, slice));
    sorter.sort(Kokkos::subview(ux3, slice));
    sorter.sort(Kokkos::subview(weight, slice));
    sorter.sort(Kokkos::subview(tag, slice));
    if constexpr (D == Dim::_2D and C != Coord::Cartesian) {
      sorter.sort(Kokkos::subview(phi, slice));
    }
    for (auto pldr { 0u }; pldr < npld_r(); ++pldr) {
      sorter.sort(Kokkos::subview(pld_r, slice, pldr));
    }
    for (auto pldi { 0u }; pldi < npld_i(); ++pldi) {
      sorter.sort(Kokkos::subview(pld_i, slice, pldi));
    }
  }

#define PARTICLES_SORT(D, C)                                                   \
  template auto Particles<D, C>::NpartsPerTagAndOffsets() const                \
    -> std::pair<std::vector<npart_t>, array_t<npart_t*>>;                     \
  template void Particles<D, C>::RemoveDead();                                 \
  template void Particles<D, C>::SortSpatially(const Grid<D>&);

  PARTICLES_SORT(Dim::_1D, Coord::Cartesian)
  PARTICLES_SORT(Dim::_2D, Coord::Cartesian)
  PARTICLES_SORT(Dim::_3D, Coord::Cartesian)
  PARTICLES_SORT(Dim::_2D, Coord::Spherical)
  PARTICLES_SORT(Dim::_2D, Coord::Qspherical)
  PARTICLES_SORT(Dim::_3D, Coord::Spherical)
  PARTICLES_SORT(Dim::_3D, Coord::Qspherical)
#undef PARTICLES_SORT

} // namespace ntt

// template <Dimension D, typename T>
// void AllocateArrayOnGrid(nddata_t<D, T>&    arr,
//                          const Grid<D>&     grid,
//                          const std::string& name) {
//   if constexpr (D == Dim::_1D) {
//     arr = nddata_t<D, T> { name, grid.n_active(in::x1) };
//   } else if constexpr (D == Dim::_2D) {
//     arr = nddata_t<D, T> { name, grid.n_active(in::x1), grid.n_active(in::x2) };
//   } else if constexpr (D == Dim::_3D) {
//     arr = nddata_t<D, T> { name,
//                            grid.n_active(in::x1),
//                            grid.n_active(in::x2),
//                            grid.n_active(in::x3) };
//   } else {
//     raise::Error("Unsupported dimension for array allocation", HERE);
//   }
// }
//
//
// array_t<ncells_t*> cell_idx { "cell_indices", npart() };
// const auto         num_cells = grid.num_active();
//
// nddata_t<D, npart_t> num_ppc;
// nddata_t<D, npart_t> disp_map;
// AllocateArrayOnGrid<D, npart_t>(num_ppc, grid, "num_ppc");
// AllocateArrayOnGrid<D, npart_t>(disp_map, grid, "disp_map");
// auto num_ppc_scatter =
// Kokkos::Experimental::create_scatter_view(num_ppc); Kokkos::parallel_for(
//   "ComputeNumPPC",
//   rangeActiveParticles(),
//   Lambda(index_t p) {
//     if (tag_p(p) != ParticleTag::alive) {
//       return;
//     }
//     auto num_ppc_acc = num_ppc_scatter.access();
//     if constexpr (D == Dim::_1D) {
//       num_ppc_acc(i1_p(p)) += 1u;
//     } else if constexpr (D == Dim::_2D) {
//       num_ppc_acc(i1_p(p), i2_p(p)) += 1u;
//     } else {
//       num_ppc_acc(i1_p(p), i2_p(p), i3_p(p)) += 1u;
//     }
//   });
// Kokkos::Experimental::contribute(num_ppc, num_ppc_scatter);
//
// npart_t  total_sum   = 0u;
// Kokkos::parallel_scan(
//   "ComputeDisplacementMap",
//   total_cells,
//   Lambda(index_t cell, npart_t & cumulative_sum, bool is_final) {
//     ncells_t i1, i2, i3;
//     if constexpr (D == Dim::_1D) {
//       i1 = cell;
//     } else if constexpr (D == Dim::_2D) {
//       i1 = cell / nx2;
//       i2 = cell % nx2;
//     } else {
//       i1 = cell / (nx2 * nx3);
//       i2 = (cell % (nx2 * nx3)) / nx3;
//       i3 = cell % nx3;
//     }
//     if (is_final) {
//       if constexpr (D == Dim::_1D) {
//         disp_map(i1) = cumulative_sum;
//       } else if constexpr (D == Dim::_2D) {
//         disp_map(i1, i2) = cumulative_sum;
//       } else {
//         disp_map(i1, i2, i3) = cumulative_sum;
//       }
//     }
//     if constexpr (D == Dim::_1D) {
//       cumulative_sum += num_ppc(i1);
//     } else if constexpr (D == Dim::_2D) {
//       cumulative_sum += num_ppc(i1, i2);
//     } else {
//       cumulative_sum += num_ppc(i1, i2, i3);
//     }
//   },
//   total_sum);
