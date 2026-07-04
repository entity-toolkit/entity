/**
 * @file utils/sorting.h
 * @brief Defines sorting functors for particle sorting
 * @implements
 *   - sort::BinBool<>
 *   - sort::BinTag<>
 *   - sort::PositionToTileIndex<>
 *   - sort::backend tag types (compile-time tag dispatch for sort_by_key)
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

  /**
   * @brief Bin a particle into a tile of edge length `tile_size` cells.
   * @tparam D     Dimension.
   * @tparam Count If true, atomic-increment `num_ppt[tile]` for each live
   *               particle (used to populate per-tile counts in one pass).
   * @tparam UsePrev If true, the bin key uses `min(i_curr, i_prev)` instead
   *                 of `i_curr` alone. This guarantees that a particle whose
   *                 Esirkepov stencil straddles a tile boundary (because it
   *                 crossed the boundary during the pusher) lands in the
   *                 lower-indexed of the two tiles. Combined with a halo of
   *                 `O+1` cells in the deposit's per-tile scratch, this
   *                 keeps every particle's stencil inside its assigned
   *                 tile's interior+halo region. See plan §S2.4.
   *
   * Dead particles get the sentinel `total_tiles + 1u` so they sort to the
   * end (or get skipped, depending on the consumer).
   */
  template <Dimension D, bool Count, bool UsePrev = false>
  struct PositionToTileIndex {
    const array_t<int*>   i1, i2, i3;
    const array_t<int*>   i1_prev, i2_prev, i3_prev;
    const array_t<short*> tag;
    array_t<ncells_t*>    tile_indices;
    ncells_t              tile_size;
    array_t<npart_t*>     num_ppt;

    ncells_t ntx2 { 0u }, ntx3 { 0u };
    ncells_t total_tiles { 0u };
    // Active-cell extents per axis. Used to clamp the bin key when
    // UsePrev=true, since `i_prev` can be transiently negative after the
    // pusher's periodic wrap (`i_prev -= ni`) or out-of-range after an
    // MPI receive that hasn't translated frames. Without clamping, the
    // signed-to-unsigned promotion in `int(-1) / uint32_t(T)` produces
    // ~1.07e9, the linearised `tile_indices(p)` overflows past `n_bins`,
    // and BinSort's internal `atomic_add(&bin_count[wild_idx], 1)`
    // faults on an unmapped page.
    int ncells1 { 1 }, ncells2 { 1 }, ncells3 { 1 };

    PositionToTileIndex(const array_t<int*>&         i1_,
                        const array_t<int*>&         i2_,
                        const array_t<int*>&         i3_,
                        const array_t<short*>&       tag_,
                        array_t<ncells_t*>&          tile_indices_,
                        const std::vector<ncells_t>& ncells,
                        ncells_t                     tile_size_ = 1u,
                        const array_t<npart_t*>& num_ppt_ = { "num_ppt", 0u },
                        const array_t<int*>& i1_prev_ = {},
                        const array_t<int*>& i2_prev_ = {},
                        const array_t<int*>& i3_prev_ = {})
      : i1 { i1_ }
      , i2 { i2_ }
      , i3 { i3_ }
      , i1_prev { i1_prev_ }
      , i2_prev { i2_prev_ }
      , i3_prev { i3_prev_ }
      , tag { tag_ }
      , tile_indices { tile_indices_ }
      , tile_size { tile_size_ }
      , num_ppt { num_ppt_ }
      , ntx2 { 1u }
      , ntx3 { 1u }
      , total_tiles { 1u } {
      raise::ErrorIf(ncells.size() < static_cast<std::size_t>(D),
                     "ncells size must match D",
                     HERE);
      if constexpr ((D == Dim::_1D) or (D == Dim::_2D) or (D == Dim::_3D)) {
        ncells1       = static_cast<int>(ncells[0]);
        npart_t ntx1  = static_cast<ncells_t>(math::ceil(
          static_cast<double>(ncells[0]) / static_cast<double>(tile_size)));
        total_tiles  *= ntx1;
      }
      if constexpr ((D == Dim::_2D) or (D == Dim::_3D)) {
        ncells2      = static_cast<int>(ncells[1]);
        ntx2         = static_cast<ncells_t>(math::ceil(
          static_cast<double>(ncells[1]) / static_cast<double>(tile_size)));
        total_tiles *= ntx2;
      }
      if constexpr (D == Dim::_3D) {
        ncells3      = static_cast<int>(ncells[2]);
        ntx3         = static_cast<ncells_t>(math::ceil(
          static_cast<double>(ncells[2]) / static_cast<double>(tile_size)));
        total_tiles *= ntx3;
      }
      if constexpr (Count) {
        raise::ErrorIf(num_ppt.extent(0) != total_tiles,
                       "num_ppt must have extent equal to total tiles",
                       HERE);
      }
      if constexpr (UsePrev) {
        raise::ErrorIf(
          i1_prev.extent(0) == 0u,
          "PositionToTileIndex<UsePrev=true> requires i1_prev to be set",
          HERE);
        if constexpr ((D == Dim::_2D) or (D == Dim::_3D)) {
          raise::ErrorIf(
            i2_prev.extent(0) == 0u,
            "PositionToTileIndex<UsePrev=true> requires i2_prev to be set",
            HERE);
        }
        if constexpr (D == Dim::_3D) {
          raise::ErrorIf(
            i3_prev.extent(0) == 0u,
            "PositionToTileIndex<UsePrev=true> requires i3_prev to be set",
            HERE);
        }
      }
    }

    Inline auto operator()(prtlidx_t p) const {
      if (tag(p) != ntt::ParticleTag::alive) {
        tile_indices(p) = total_tiles + 1u;
      } else {
        // bin key per-axis: use min(i, i_prev) when UsePrev so that a
        // particle straddling a boundary lands in the lower tile.
        // Then clamp to [0, ncells_axis - 1] — `i_prev` can be negative
        // (after the pusher's periodic-wrap path: `i_prev -= ni`) or
        // out-of-range (after MPI receive without frame translation).
        // Without the clamp, signed-to-unsigned promotion in
        // `int(-1) / uint32_t(T)` makes `tile_indices(p)` overflow far
        // past `n_bins`, and BinSort's `atomic_add(&bin_count[bin],1)`
        // faults on an unmapped page.
        const auto clamp_axis = [](int v, int ncells) -> int {
          return (v < 0) ? 0 : ((v >= ncells) ? (ncells - 1) : v);
        };
        const auto key1 = [&]() -> int {
          if constexpr (UsePrev) {
            const int raw = (i1(p) < i1_prev(p)) ? i1(p) : i1_prev(p);
            return clamp_axis(raw, ncells1);
          } else {
            return i1(p);
          }
        }();
        const auto key2 = [&]() -> int {
          if constexpr (UsePrev) {
            const int raw = (i2(p) < i2_prev(p)) ? i2(p) : i2_prev(p);
            return clamp_axis(raw, ncells2);
          } else {
            return i2(p);
          }
        }();
        const auto key3 = [&]() -> int {
          if constexpr (UsePrev) {
            const int raw = (i3(p) < i3_prev(p)) ? i3(p) : i3_prev(p);
            return clamp_axis(raw, ncells3);
          } else {
            return i3(p);
          }
        }();
        if constexpr (D == Dim::_1D) {
          tile_indices(p) = static_cast<ncells_t>(key1 / tile_size);
        } else if constexpr (D == Dim::_2D) {
          tile_indices(p) = static_cast<ncells_t>(key1 / tile_size) * ntx2 +
                            static_cast<ncells_t>(key2 / tile_size);
        } else if constexpr (D == Dim::_3D) {
          tile_indices(p) = (static_cast<ncells_t>(key1 / tile_size) * ntx2 +
                             static_cast<ncells_t>(key2 / tile_size)) *
                              ntx3 +
                            static_cast<ncells_t>(key3 / tile_size);
        } else {
          raise::KernelError(HERE, "Wrong D in SortSpatially");
        }
        if constexpr (Count) {
          Kokkos::atomic_add(&num_ppt(tile_indices(p)), 1);
        }
      }
    }
  };

  // -------------------- Backend dispatch for sort_by_key ------------------- //
  // Compile-time tags for tag-dispatch into backend-specific
  // sort_by_key implementations. Selection is fully compile-time: the
  // backend that resolves depends on the active Kokkos device and the
  // availability of the corresponding vendor library.
  namespace backend {
    struct OneDPL {};
    struct Thrust {};
    struct Rocthrust {};
    struct StdSort {};
    // Always-available legacy fallback using Kokkos::BinSort.
    struct BinSort {};
  } // namespace backend

} // namespace sort

#endif // GLOBAL_UTILS_SORTING_H
