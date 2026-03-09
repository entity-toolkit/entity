/**
 * @file utils/sorting.h
 * @brief Comparator structs for Kokkos bin-sorting
 * @implements
 *   - sort::BinBool<>
 *   - sort::BinTag<>
 *   - sort::PositionToCellIndex<>
 * @namespaces:
 *   - sort::
 * @note BinBool sorts by boolean values "true" then "false"
 * @note BinTag sorts by tag values "1" then "0" then "2" ... "n"
 */

#ifndef GLOBAL_UTILS_SORTING_H
#define GLOBAL_UTILS_SORTING_H

#include "enums.h"
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

  template <Dimension D>
  struct PositionToCellIndex {
    const array_t<int*>   i1, i2, i3;
    const array_t<short*> tag;
    array_t<ncells_t*>    cell_indices;
    ncells_t              nx2, nx3, total_cells;

    PositionToCellIndex(const array_t<int*>&   i1,
                        const array_t<int*>&   i2,
                        const array_t<int*>&   i3,
                        const array_t<short*>& tag,
                        array_t<ncells_t*>&    cell_indices,
                        ncells_t               nx2,
                        ncells_t               nx3,
                        ncells_t               total_cells)
      : i1 { i1 }
      , i2 { i2 }
      , i3 { i3 }
      , tag { tag }
      , cell_indices { cell_indices }
      , nx2 { nx2 }
      , nx3 { nx3 }
      , total_cells { total_cells } {}

    Inline auto operator()(index_t p) const {
      if (tag(p) != ntt::ParticleTag::alive) {
        cell_indices(p) = total_cells + 1u;
      } else {
        if constexpr (D == Dim::_1D) {
          cell_indices(p) = static_cast<ncells_t>(i1(p));
        } else if constexpr (D == Dim::_2D) {
          cell_indices(p) = static_cast<ncells_t>(i1(p)) * nx2 +
                            static_cast<ncells_t>(i2(p));
        } else if constexpr (D == Dim::_3D) {
          cell_indices(p) = (static_cast<ncells_t>(i1(p)) * nx2 +
                             static_cast<ncells_t>(i2(p))) *
                              nx3 +
                            static_cast<ncells_t>(i3(p));
        } else {
          raise::KernelError(HERE, "Wrong D in SortSpatially");
        }
      }
    }
  };

} // namespace sort

#endif // GLOBAL_UTILS_SORTING_H
