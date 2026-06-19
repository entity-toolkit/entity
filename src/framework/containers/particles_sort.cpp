#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/sorting.h"

#include "framework/containers/particles.h"
#include "framework/domain/grid.h"

#if defined(TEAM_POLICY)
  #if (defined(SYCL_ENABLED) && defined(ONEDPL_ENABLED)) ||                    \
      (defined(CUDA_ENABLED) && defined(THRUST_ENABLED)) ||                    \
      (defined(HIP_ENABLED) && defined(ROCTHRUST_ENABLED))
    #define TEAM_POLICY_USE_VENDOR_SORT
    #include "utils/sort_dispatch.h"
  #endif
#endif

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <algorithm>
#include <cstddef>
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

    // count # of particles per each tag, skipping the alive bin in-kernel.
    constexpr short tag_alive_s = static_cast<short>(ParticleTag::alive);
    Kokkos::parallel_for(
      "NpartPerTag",
      rangeActiveParticles(),
      Lambda(prtlidx_t p) {
        const short t = this_tag(p);
        if (t < 0 || t >= static_cast<short>(num_tags)) {
          raise::KernelError(HERE, "Invalid tag value");
        }
        if (t != tag_alive_s) {
          Kokkos::atomic_add(&npptag(t), static_cast<npart_t>(1));
        }
      });

    // copy the count to a vector on the host and reconstruct the alive bin
    auto npptag_h = Kokkos::create_mirror_view(npptag);
    Kokkos::deep_copy(npptag_h, npptag);
    std::vector<npart_t> npptag_vec(num_tags);
    npart_t              non_alive_total = 0;
    for (auto t { 0u }; t < num_tags; ++t) {
      npptag_vec[t] = npptag_h(t);
      if (static_cast<short>(t) != tag_alive_s) {
        non_alive_total += npptag_h(t);
      }
    }
    npptag_vec[tag_alive_s] = npart() - non_alive_total;

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
      Lambda(prtlidx_t p) { buffer(p) = arr(indices_alive(p)); });

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
      CreateParticleRangePolicy<Dim::_2D>(
        { 0, 0 },
        { n_alive, static_cast<npart_t>(arr.extent(1)) }),
      Lambda(prtlidx_t p, prtlidx_t l) {
        buffer(p, l) = arr(indices_alive(p), l);
      });

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
      Lambda(prtlidx_t p, npart_t & nalive, npart_t & ndead) {
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
      Lambda(prtlidx_t p) {
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
#if defined(TEAM_POLICY)
    // ---------------------- team_policy: tile-based sort ------------------ //
    const auto npart_local = npart();
    if (npart_local == 0u) {
      m_tile_layout = TileLayout<D> {};
      m_is_sorted   = true;
      return;
    }

    constexpr unsigned short T = static_cast<unsigned short>(
      TEAM_POLICY_TILE_SIZE);
    static_assert(T > 0u, "TEAM_POLICY_TILE_SIZE must be > 0");

    // 1. Compute per-axis tile counts and total_tiles.
    const auto ncells_active = grid.n_active();
    ncells_t   ntx[3] { 1u, 1u, 1u };
    ncells_t   total_tiles { 1u };
    if constexpr ((D == Dim::_1D) or (D == Dim::_2D) or (D == Dim::_3D)) {
      ntx[0]       = static_cast<ncells_t>(math::ceil(
        static_cast<double>(ncells_active[0]) / static_cast<double>(T)));
      total_tiles *= ntx[0];
    }
    if constexpr ((D == Dim::_2D) or (D == Dim::_3D)) {
      ntx[1]       = static_cast<ncells_t>(math::ceil(
        static_cast<double>(ncells_active[1]) / static_cast<double>(T)));
      total_tiles *= ntx[1];
    }
    if constexpr (D == Dim::_3D) {
      ntx[2]       = static_cast<ncells_t>(math::ceil(
        static_cast<double>(ncells_active[2]) / static_cast<double>(T)));
      total_tiles *= ntx[2];
    }

    // 2. Compute per-particle tile key (with min(i, i_prev)).
    array_t<ncells_t*> tile_indices { "tile_indices", npart_local };
    Kokkos::parallel_for(
      "FillTileIndices",
      rangeActiveParticles(),
      sort::PositionToTileIndex<D, false, true> { i1,
                                                   i2,
                                                   i3,
                                                   tag,
                                                   tile_indices,
                                                   ncells_active,
                                                   static_cast<ncells_t>(T),
                                                   array_t<npart_t*> {},
                                                   i1_prev,
                                                   i2_prev,
                                                   i3_prev });

    // 3. Sort. Vendor library (oneDPL/Thrust) when compiled in;
    //    Kokkos::BinSort otherwise. n_bins = total_tiles + 2 covers
    //    the dead-particle sentinel bin (total_tiles + 1u).
    const ncells_t n_bins = total_tiles + 2u;
    const auto     slice  = prtl_slice_t(0, npart_local);
  #if defined(TEAM_POLICY_USE_VENDOR_SORT)
    // Vendor path: produce an explicit permutation via sort_by_key,
    // then apply it to each SoA member by gathering into a fresh
    // full-capacity buffer and swapping the View handle in (no
    // copy-back). The *_prev arrays are skipped — see
    // apply_permutation_to_soa. Peak transient = one
    // `maxnpart × sizeof(member)` buffer at a time.
    prtl_perm_t perm { "tile_perm", npart_local };
    #if defined(SYCL_ENABLED) && defined(ONEDPL_ENABLED)
    sort_helpers::sort_by_key_dispatch(tile_indices,
                                       perm,
                                       n_bins,
                                       sort::backend::OneDPL {});
    #elif defined(HIP_ENABLED) && defined(ROCTHRUST_ENABLED)
    sort_helpers::sort_by_key_dispatch(tile_indices,
                                       perm,
                                       n_bins,
                                       sort::backend::Rocthrust {});
    #else
    sort_helpers::sort_by_key_dispatch(tile_indices,
                                       perm,
                                       n_bins,
                                       sort::backend::Thrust {});
    #endif
    Kokkos::fence("SortSpatially: pre-gather drain");
    apply_permutation_to_soa(perm);
  #else
    // BinSort path: same mechanism as legacy SortSpatially (BinSort
    // allocates one temp View per `sorter.sort(view)` call and frees
    // it before the next), so peak transient memory is bounded.
    using sorter_op_t = Kokkos::BinOp1D<array_t<ncells_t*>>;
    using sorter_t    = Kokkos::BinSort<array_t<ncells_t*>, sorter_op_t>;
    auto bin_op       = sorter_op_t { static_cast<int>(n_bins), 0u, n_bins };
    auto sorter       = sorter_t { tile_indices, bin_op, false };
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
    // Apply the same permutation to `tile_indices` itself so it ends
    // monotonically non-decreasing for the offsets pass below.
    sorter.sort(tile_indices);
  #endif // TEAM_POLICY_USE_VENDOR_SORT

    // 5. Compute per-tile prefix-sum `tile_offsets` for the tiled
    //    pusher. `tile_indices` is now sorted (monotonically
    //    non-decreasing for alive particles, dead sentinel
    //    `total_tiles + 1` clustered at the end) — vendor sort_by_key
    //    sorts keys in place; the BinSort path explicitly applies the
    //    same permutation to `tile_indices` above. Transition-detect
    //    directly on it: the start of each non-empty tile is the only
    //    place a write happens — atomic-free in the dense branch.
    //    Empty tiles (no particles) are filled by a reverse pass on a
    //    small host mirror (`total_tiles ≈ 176K` at production scale →
    //    ~700 KB).
    {
      array_t<npart_t*> tile_offsets { "tile_offsets", total_tiles + 1u };
      Kokkos::deep_copy(tile_offsets, static_cast<npart_t>(npart_local));

      const auto total_tiles_v = total_tiles;
      auto       ti_v          = tile_indices;
      Kokkos::parallel_for(
        "DetectTileBoundaries",
        rangeActiveParticles(),
        Lambda(prtlidx_t p) {
          const auto t_curr   = ti_v(p);
          const bool boundary = (p == 0u) || (ti_v(p - 1u) != t_curr);
          if (!boundary) {
            return;
          }
          if (t_curr < total_tiles_v) {
            tile_offsets(t_curr) = p;
          } else {
            // First dead particle — also marks the alive_count boundary
            // stored at index total_tiles.
            Kokkos::atomic_min(&tile_offsets(total_tiles_v), p);
          }
        });

      auto h_offsets = Kokkos::create_mirror_view(tile_offsets);
      Kokkos::deep_copy(h_offsets, tile_offsets);
      for (auto t = static_cast<std::size_t>(total_tiles); t-- > 0u;) {
        if (h_offsets(t) > h_offsets(t + 1u)) {
          h_offsets(t) = h_offsets(t + 1u);
        }
      }
      Kokkos::deep_copy(tile_offsets, h_offsets);

      m_tile_layout.tile_offsets = tile_offsets;
      // tile_offsets(total_tiles) is the alive-particle count at sort time:
      // the tiles partition exactly [0, npart_partitioned). The deposit
      // launcher compares this against the live npart() to detect (and
      // separately deposit) particles appended since this sort.
      m_tile_layout.npart_partitioned = h_offsets(total_tiles);
    }

    // 6. Populate `m_tile_layout` size/shape. `tile_perm` is not used
    //    in the current design — the SoA arrays are physically permuted
    //    into tile order, so consumers iterate
    //    `[tile_offsets(t), tile_offsets(t+1))` directly without a
    //    separate permutation indirection.
    m_tile_layout.ntiles_per_axis[0] = ntx[0];
    m_tile_layout.ntiles_per_axis[1] = ntx[1];
    m_tile_layout.ntiles_per_axis[2] = ntx[2];
    m_tile_layout.ntiles_total       = total_tiles;
    m_tile_layout.tile_size          = T;
    m_tile_layout.tile_perm          = prtl_perm_t {};
    m_is_sorted                      = true;

    Kokkos::fence("SortSpatially: end of team_policy path");
#else  // !TEAM_POLICY — legacy in-place BinSort by global cell index
    const auto nx2         = grid.n_active(in::x2);
    const auto nx3         = grid.n_active(in::x3);
    const auto total_cells = grid.num_active();

    array_t<ncells_t*> cell_indices { "cell_indices", npart() };

    Kokkos::parallel_for("FillCellIndices",
                         rangeActiveParticles(),
                         sort::PositionToTileIndex<D, false> { i1,
                                                               i2,
                                                               i3,
                                                               tag,
                                                               cell_indices,
                                                               grid.n_active() });
    const auto slice = prtl_slice_t(0, npart());

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
#endif // TEAM_POLICY
  }

#if defined(TEAM_POLICY_USE_VENDOR_SORT)
  namespace permute_helpers {

    // Permute a 1D SoA member array `arr` by `perm`. Gathers into a
    // fresh buffer allocated at the member's full capacity (maxnpart),
    // then swaps the View handle in. This avoids the redundant copy-back
    // pass of the old gather-then-deep_copy approach (~2x less HBM
    // traffic). Allocating at full capacity preserves the member's spare
    // room for injection; the untouched tail [n, capacity) is
    // zero-initialized by Kokkos (cleaner than the stale values the old
    // deep_copy left there). The fence drains the gather (which reads the
    // old storage) before the swap drops the last reference to it.
    template <typename V>
    inline void permute_1d_swap(V&                 arr,
                                const prtl_perm_t& perm,
                                npart_t            n) {
      if (n == 0u) {
        return;
      }
      V    buf(arr.label(), arr.extent(0));
      auto perm_v = perm;
      auto arr_v  = arr;
      Kokkos::parallel_for(
        "Permute1D",
        n,
        KOKKOS_LAMBDA(const npart_t p) { buf(p) = arr_v(perm_v(p)); });
      Kokkos::fence("permute_1d_swap: end");
      arr = buf;
    }

    // 2D analogue for `pld_r` / `pld_i`.
    template <typename V>
    inline void permute_2d_swap(V&                 arr,
                                const prtl_perm_t& perm,
                                npart_t            n,
                                npart_t            ncols) {
      if (n == 0u or ncols == 0u) {
        return;
      }
      V    buf(arr.label(), arr.extent(0), arr.extent(1));
      auto perm_v = perm;
      auto arr_v  = arr;
      Kokkos::parallel_for(
        "Permute2D",
        CreateParticleRangePolicy<Dim::_2D>({ 0u, 0u }, { n, ncols }),
        KOKKOS_LAMBDA(const npart_t p, const npart_t l) {
          buf(p, l) = arr_v(perm_v(p), l);
        });
      Kokkos::fence("permute_2d_swap: end");
      arr = buf;
    }

  } // namespace permute_helpers

  template <Dimension D, Coord::type C>
  void Particles<D, C>::apply_permutation_to_soa(const prtl_perm_t& perm) {
    const auto n = npart();
    if (n == 0u) {
      return;
    }

    using permute_helpers::permute_1d_swap;
    using permute_helpers::permute_2d_swap;

    // The *_prev arrays (i{1,2,3}_prev, dx{1,2,3}_prev) are intentionally
    // NOT permuted. SortSpatially runs at the very end of the step loop
    // (engine step_forward), and the first thing the next step's pusher
    // does is overwrite prev := current (positionPush, sr.hpp / gr.hpp)
    // for every active particle, before any consumer reads prev:
    //   - current deposit: runs after the push, which has already
    //     overwritten prev; species with pusher==NONE (whose prev would
    //     stay un-permuted) are skipped by CurrentsDeposit entirely.
    //   - pusher getParticlePrevPosition / piston: read prev only after
    //     positionPush has rewritten it within the same call.
    //   - checkpoint (prev is checkpoint-only, never in diagnostic
    //     output): on restart the first push overwrites prev before it
    //     is read, so restart results are unaffected; only the redundant
    //     prev field saved to the checkpoint differs from the old code.
    // Permuting prev would therefore reorder data that is overwritten
    // before it is ever observed.
    if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
      permute_1d_swap(i1, perm, n);
      permute_1d_swap(dx1, perm, n);
    }
    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      permute_1d_swap(i2, perm, n);
      permute_1d_swap(dx2, perm, n);
    }
    if constexpr (D == Dim::_3D) {
      permute_1d_swap(i3, perm, n);
      permute_1d_swap(dx3, perm, n);
    }
    permute_1d_swap(ux1, perm, n);
    permute_1d_swap(ux2, perm, n);
    permute_1d_swap(ux3, perm, n);
    permute_1d_swap(weight, perm, n);
    permute_1d_swap(tag, perm, n);
    if constexpr (D == Dim::_2D and C != Coord::Cartesian) {
      permute_1d_swap(phi, perm, n);
    }
    if (npld_r() > 0) {
      permute_2d_swap(pld_r, perm, n, static_cast<npart_t>(npld_r()));
    }
    if (npld_i() > 0) {
      permute_2d_swap(pld_i, perm, n, static_cast<npart_t>(npld_i()));
    }
  }
#endif // TEAM_POLICY_USE_VENDOR_SORT

#if defined(TEAM_POLICY_USE_VENDOR_SORT)
  #define APPLY_PERM_INSTANTIATE(D, C)                                         \
    template void Particles<D, C>::apply_permutation_to_soa(                   \
      const prtl_perm_t&);
#else
  #define APPLY_PERM_INSTANTIATE(D, C)
#endif

#define PARTICLES_SORT(D, C)                                                   \
  template auto Particles<D, C>::NpartsPerTagAndOffsets() const                \
    -> std::pair<std::vector<npart_t>, array_t<npart_t*>>;                     \
  template void Particles<D, C>::RemoveDead();                                 \
  template void Particles<D, C>::SortSpatially(const Grid<D>&);                \
  APPLY_PERM_INSTANTIATE(D, C)

  PARTICLES_SORT(Dim::_1D, Coord::Cartesian)
  PARTICLES_SORT(Dim::_2D, Coord::Cartesian)
  PARTICLES_SORT(Dim::_3D, Coord::Cartesian)
  PARTICLES_SORT(Dim::_2D, Coord::Spherical)
  PARTICLES_SORT(Dim::_2D, Coord::Qspherical)
  PARTICLES_SORT(Dim::_3D, Coord::Spherical)
  PARTICLES_SORT(Dim::_3D, Coord::Qspherical)
#undef PARTICLES_SORT
#undef APPLY_PERM_INSTANTIATE

} // namespace ntt
