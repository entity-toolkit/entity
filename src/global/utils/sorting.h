/**
 * @file utils/sorting.h
 * @brief Comparator structs for Kokkos bin-sorting
 * @implements
 *   - sort::BinBool<>
 *   - sort::BinTag<>
 *   - sort::PositionToTileIndex<>
 * @namespaces:
 *   - sort::
 * @note BinBool sorts by boolean values "true" then "false"
 * @note BinTag sorts by tag values "1" then "0" then "2" ... "n"
 */

#ifndef GLOBAL_UTILS_SORTING_H
#define GLOBAL_UTILS_SORTING_H

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

namespace sort {

  template <class KeyViewType>
  struct BinBool {
    BinBool() = default;

    template <class ViewType>
    Inline auto bin(ViewType& keys, int i) const -> int {
      return keys(i) ? 1 : 0;
    }

    Inline auto max_bins() const -> int {
      return 2;
    }

    template <class ViewType, typename iT1, typename iT2>
    Inline auto operator()(ViewType&, iT1&, iT2&) const -> bool {
      return false;
    }
  };

  template <class KeyViewType>
  struct BinTag {
    BinTag(int max_bins) : m_max_bins { max_bins } {}

    template <class ViewType>
    Inline auto bin(ViewType& keys, int i) const -> int {
      return (keys(i) == 0) ? 1 : ((keys(i) == 1) ? 0 : keys(i));
    }

    Inline auto max_bins() const -> int {
      return m_max_bins;
    }

    template <class ViewType, typename iT1, typename iT2>
    Inline auto operator()(ViewType&, iT1&, iT2&) const -> bool {
      return false;
    }

  private:
    const int m_max_bins;
  };

  template <Dimension D, bool Count>
  struct PositionToTileIndex {
    const array_t<int*>   i1, i2, i3;
    const array_t<short*> tag;
    array_t<ncells_t*>    tile_indices;
    ncells_t              tile_size;
    array_t<npart_t*>     num_ppt;

    ncells_t ntx1, ntx2;
    ncells_t total_tiles;

    PositionToTileIndex(const array_t<int*>&        i1,
                        const array_t<int*>&        i2,
                        const array_t<int*>&        i3,
                        const array_t<short*>&      tag,
                        array_t<ncells_t*>&         tile_indices,
                        const std::vector<ncells_t> ncells,
                        ncells_t                    tile_size = 1u,
                        array_t<npart_t*>           num_ppt = { "num_ppt", 0u })
      : i1 { i1 }
      , i2 { i2 }
      , i3 { i3 }
      , tag { tag }
      , tile_indices { tile_indices }
      , tile_size { tile_size }
      , num_ppt { num_ppt } {
      raise::ErrorIf(ncells.size() < static_cast<std::size_t>(D),
                     "ncells size must match D",
                     HERE);
      ntx1          = static_cast<ncells_t>(math::ceil(
        static_cast<double>(ncells[0]) / static_cast<double>(tile_size)));
      ntx2          = 1u;
      ncells_t ntx3 = 1u;
      if constexpr ((D == Dim::_2D) or (D == Dim::_3D)) {
        ntx2 = static_cast<ncells_t>(math::ceil(
          static_cast<double>(ncells[1]) / static_cast<double>(tile_size)));
      }
      if constexpr (D == Dim::_3D) {
        ntx3 = static_cast<ncells_t>(math::ceil(
          static_cast<double>(ncells[2]) / static_cast<double>(tile_size)));
      }
      total_tiles = ntx1 * ntx2 * ntx3;
      if constexpr (Count) {
        raise::ErrorIf(num_ppt.extent(0) != total_tiles,
                       "num_ppt must have extent equal to total tiles",
                       HERE);
      }
    }

    Inline auto operator()(index_t p) const {
      if (tag(p) != ntt::ParticleTag::alive) {
        tile_indices(p) = total_tiles + 1u;
      } else {
        if constexpr (D == Dim::_1D) {
          tile_indices(p) = static_cast<ncells_t>(i1(p) / tile_size);
        } else if constexpr (D == Dim::_2D) {
          tile_indices(p) = static_cast<ncells_t>(i2(p) / tile_size) * ntx1 +
                            static_cast<ncells_t>(i1(p) / tile_size);
        } else if constexpr (D == Dim::_3D) {
          tile_indices(p) = (static_cast<ncells_t>(i3(p) / tile_size) * ntx2 +
                             static_cast<ncells_t>(i2(p) / tile_size)) *
                              ntx1 +
                            static_cast<ncells_t>(i1(p) / tile_size);
        } else {
          raise::KernelError(HERE, "Wrong D in SortSpatially");
        }
        if constexpr (Count) {
          Kokkos::atomic_add(&num_ppt(tile_indices(p)), 1);
        }
      }
    }
  };

} // namespace sort

#endif // GLOBAL_UTILS_SORTING_H
