/**
 * @file kernels/tiled_scatter.hpp
 * @brief Generic tiled (TeamPolicy + per-team SLM scratch) particle->grid
 *        scatter harness, shared by the current deposit and the hybrid
 *        moment deposit.
 *
 * The harness owns everything that is NOT per-particle math: tile-id
 * decode, scratch allocation/zero, particle-slice clamping, the
 * scratch-vs-global routing (per-particle escape valve), the team
 * barrier, and the bounds-clipped cooperative flush to the global field.
 * The per-particle physics lives in a `Body` functor supplied by the
 * caller; flat and tiled kernels of one deposit differ only in the sink
 * their body writes through.
 *
 * @implements
 *   - kernel::TiledScatter_kernel<>   (TEAM_POLICY only)
 *   - kernel::MakeTiledPolicy<>       (TEAM_POLICY only, host)
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_TILED_SCATTER_HPP
#define KERNELS_TILED_SCATTER_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include <Kokkos_Core.hpp>

#if defined(TEAM_POLICY)

namespace kernel {
  using namespace ntt;

  /**
   * @brief Generic tiled particle->grid scatter harness.
   *
   * One team per spatial tile (`league_size = ntiles_total`). Each team
   * accumulates particle contributions into a per-team scratch buffer of
   * shape `(T_TILE + 2*HALO)^D × NC` real_t, where `HALO = REACH + DRIFT`
   * cells per side. Scratch atomics live in SLM (PVC: ~5–10 cycles per
   * `atomic_add`); the global field is touched only once per scratch cell
   * at flush time. Compared with a flat scatter-view kernel:
   *   - global atomic pressure ~ (T_TILE + 2*HALO)^D × NC per tile
   *     instead of (stencil writes per particle × particles)
   *   - per-particle stencil writes are tile-local (SLM) instead of
   *     scattering through global HBM
   *
   *   D       : dimension (must match the field view)
   *   NC      : number of scratch components accumulated per cell
   *   NG      : component count of the global field view (NC <= NG)
   *   REACH   : one-sided stencil reach in cells, used ONLY for halo
   *             sizing: HALO = REACH + DRIFT (DRIFT = TEAM_POLICY_DRIFT,
   *             default 1)
   *   T_TILE  : tile edge length (TEAM_POLICY_TILE_SIZE)
   *   Body    : per-particle functor, see contract below
   *
   * **Body contract** (device-callable, captured by value):
   *   `template <class Sink> Inline void operator()(prtlidx_t p, Sink& sink) const;`
   * The body must, for each particle:
   *   1) decide its deposit-time footprint [lo, hi] per axis in GLOBAL
   *      field coordinates (including N_GHOSTS) and call
   *      `sink.select(lo1, hi1, ...)`; after select(), writes are routed
   *      to SLM scratch iff the whole footprint fits the tile window;
   *   2) emit contributions via `sink(gi..., c, val)` with c in [0, NC)
   *      and gi... the global field storage index (incl. N_GHOSTS).
   * The harness routes each write to tile scratch (no bounds test needed
   * — select() proved containment) or, on escape, to the global view with
   * a bounds clip against its storage extents. The global component
   * written is `comp_offset + c` (comp_offset = 0 for currents / hybrid
   * moments, buff_idx for single-component output moments).
   *
   * **Halo sizing and escape valve.** The scratch HALO is
   * `REACH + DRIFT`: REACH is the one-sided stencil reach the caller
   * instantiates with, DRIFT is the `team_policy_drift` CMake knob (macro
   * TEAM_POLICY_DRIFT) — the number of cells a particle may drift between
   * two sorts that the halo is sized to absorb — and `1` by default (the
   * every-step-sorted common case). It is independent of the sort
   * cadence, which is set at runtime via `spatial_sorting_interval`;
   * particles that drift past the halo take the escape valve below.
   *
   * Correctness does **not** depend on the halo size. Any particle whose
   * full footprint escapes the scratch tile — because it drifted further
   * than `DRIFT`, was reordered far from its tile by a no-sort-step
   * `CommunicateParticles`, or because the halo is otherwise undersized —
   * is deposited *as a whole* via direct, bounds-clipped
   * `Kokkos::atomic_add`s on the global field view (the per-particle
   * escape valve, driven by the body's `select()`). Each particle's
   * stencil is therefore deposited exactly once (entirely to SLM scratch
   * when it fits, entirely to global memory when it does not), so the
   * path is conservative; it is merely slower per write. Sizing `DRIFT`
   * to the typical between-sort drift keeps the common case in fast SLM;
   * sorting less often (or drifting past the halo) only costs
   * escape-valve traffic, never accuracy.
   *
   * **Partition coverage.** The team iteration covers only the particles
   * partitioned at the last sort, `[0, layout.npart_partitioned)`, clamped
   * to the live `npart`. Particles appended past the partition since the
   * sort are not seen here; the launcher deposits that tail with the
   * corresponding flat kernel so every active particle is covered exactly
   * once regardless of sort cadence.
   */
  template <Dimension D,
            unsigned short NC,
            unsigned short NG,
            int            REACH,
            unsigned short T_TILE,
            class Body>
  class TiledScatter_kernel {
    static_assert(NC > 0u and NC <= NG,
                  "TiledScatter_kernel: need 0 < NC <= NG");
    static_assert(T_TILE > 0u, "T_TILE must be positive");
    static_assert(REACH >= 0, "REACH must be non-negative");

  public:
#if defined(TEAM_POLICY_DRIFT)
    static constexpr int DRIFT = static_cast<int>(TEAM_POLICY_DRIFT);
#else
    static constexpr int DRIFT = 1;
#endif
    static constexpr int HALO = REACH + DRIFT;
    static constexpr int TE   = static_cast<int>(T_TILE) + 2 * HALO;

  private:
    static constexpr int NCi = static_cast<int>(NC);

    using exec_space  = Kokkos::DefaultExecutionSpace;
    using team_policy = Kokkos::TeamPolicy<exec_space>;
    using member_t    = typename team_policy::member_type;
    using scratch_t   = scratch_ndfield_t<D, NC, real_t>;

    ndfield_t<D, NG> F;
    Body             body;

    // Tile metadata produced by SortSpatially.
    array_t<npart_t*> tile_offsets;
    ncells_t          ntx1 { 1u }, ntx2 { 1u }, ntx3 { 1u };
    ncells_t          total_tiles { 0u };

    /**
     * Current active-particle count. `tile_offsets` partitions only the
     * particles that existed at the last sort ([0, layout.npart_partitioned));
     * `npart` may differ if a pusher dead-tagged particles in place since.
     * Each team clamps its `[tile_offsets(t), tile_offsets(t+1))` slice to
     * `npart` so stale slots past the live array are never read. Particles
     * appended *beyond* the partition (npart > npart_partitioned) are not seen
     * by any team here — the launcher deposits that tail separately.
     */
    npart_t npart { 0u };

    /**
     * F's full storage extent including all ghost cells. Used to clip the
     * cooperative flush (and the escape-valve writes) so that a partial
     * tile at the high end of the domain does not over-write past the view.
     */
    int ext1 { 0 }, ext2 { 0 }, ext3 { 0 };

    // Global component the scratch component 0 flushes into.
    int comp_offset { 0 };

  public:
    /**
     * @brief Per-particle write sink handed to the body.
     *
     * Carries the per-particle `to_scratch` routing flag, so it must be
     * constructed per particle (inside the TeamThreadRange lambda, one
     * thread per particle) and passed to the body by reference — never
     * shared across particles/threads.
     *
     * `select(lo, hi, ...)` takes the footprint in GLOBAL field storage
     * coordinates (incl. N_GHOSTS) and proves containment in the tile
     * scratch window; `operator()(gi..., c, v)` then routes the write:
     * scratch needs no per-cell bounds test (select() proved it), the
     * global escape path bounds-clips against the storage extent (writes
     * past the ghost stripe are re-supplied by SynchronizeFields; an
     * unclipped write here faults the GPU when an escaped boundary
     * particle's stencil reaches past the extent).
     */
    struct Sink {
      scratch_t        scr;
      ndfield_t<D, NG> F;
      int              o1, o2, o3; // global coord of scratch index 0 per axis
      int              e1, e2, e3; // global storage extents
      int              comp_offset;
      bool             to_scratch { false };

      Inline void select(int lo1, int hi1) {
        to_scratch = (lo1 - o1 >= 0) and (hi1 - o1 < TE);
      }

      Inline void select(int lo1, int hi1, int lo2, int hi2) {
        to_scratch = (lo1 - o1 >= 0) and (hi1 - o1 < TE) and
                     (lo2 - o2 >= 0) and (hi2 - o2 < TE);
      }

      Inline void select(int lo1, int hi1, int lo2, int hi2, int lo3, int hi3) {
        to_scratch = (lo1 - o1 >= 0) and (hi1 - o1 < TE) and
                     (lo2 - o2 >= 0) and (hi2 - o2 < TE) and
                     (lo3 - o3 >= 0) and (hi3 - o3 < TE);
      }

      Inline void operator()(int g_i1, int c, real_t v) const {
        if (to_scratch) {
          Kokkos::atomic_add(&scr(g_i1 - o1, c), v);
        } else if (g_i1 >= 0 and g_i1 < e1) {
          Kokkos::atomic_add(&F(g_i1, comp_offset + c), v);
        }
      }

      Inline void operator()(int g_i1, int g_i2, int c, real_t v) const {
        if (to_scratch) {
          Kokkos::atomic_add(&scr(g_i1 - o1, g_i2 - o2, c), v);
        } else if (g_i1 >= 0 and g_i1 < e1 and g_i2 >= 0 and g_i2 < e2) {
          Kokkos::atomic_add(&F(g_i1, g_i2, comp_offset + c), v);
        }
      }

      Inline void operator()(int g_i1, int g_i2, int g_i3, int c, real_t v) const {
        if (to_scratch) {
          Kokkos::atomic_add(&scr(g_i1 - o1, g_i2 - o2, g_i3 - o3, c), v);
        } else if (g_i1 >= 0 and g_i1 < e1 and g_i2 >= 0 and g_i2 < e2 and
                   g_i3 >= 0 and g_i3 < e3) {
          Kokkos::atomic_add(&F(g_i1, g_i2, g_i3, comp_offset + c), v);
        }
      }
    };

    TiledScatter_kernel(const ndfield_t<D, NG>& F,
                        const Body&             body,
                        const TileLayout<D>&    layout,
                        npart_t                 npart,
                        int                     comp_offset = 0)
      : F { F }
      , body { body }
      , tile_offsets { layout.tile_offsets }
      , ntx1 { layout.ntiles_per_axis[0] }
      , ntx2 { layout.ntiles_per_axis[1] }
      , ntx3 { layout.ntiles_per_axis[2] }
      , total_tiles { layout.ntiles_total }
      , npart { npart }
      , comp_offset { comp_offset } {
      raise::ErrorIf(
        layout.tile_size != T_TILE,
        "TiledScatter launched with mismatched T_TILE and runtime tile_size",
        HERE);
      raise::ErrorIf(comp_offset < 0 or
                       comp_offset + NCi > static_cast<int>(NG),
                     "TiledScatter: comp_offset + NC exceeds the field view",
                     HERE);
      /**
       * @note: HALO is allowed to exceed N_GHOSTS. The cooperative
       * scratch→F flush and the per-particle escape valve both bounds-clip
       * their writes against `ext*` so writes that would land past F's
       * ghost stripe are silently dropped (they only ever come from a
       * particle whose stencil reaches into the domain ghost region, where
       * the field synchronization will re-supply the contribution).
       */
      if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
        ext1 = static_cast<int>(F.extent(0));
      }
      if constexpr (D == Dim::_2D or D == Dim::_3D) {
        ext2 = static_cast<int>(F.extent(1));
      }
      if constexpr (D == Dim::_3D) {
        ext3 = static_cast<int>(F.extent(2));
      }
    }

    /**
     * @brief Per-team scratch size in bytes. Used by the launcher to set
     *        `team_policy.set_scratch_size(0, Kokkos::PerTeam(bytes))`.
     */
    static constexpr size_t scratch_bytes() {
      // The component count (NC) is a *static* extent of scratch_ndfield_t
      // (View<real_t*[NC]> / **[NC] / ***[NC]), so shmem_size() takes only
      // the dynamic spatial extents — passing NC as well trips Kokkos'
      // `rank_dynamic != number of arguments` abort. This matches the
      // scratch View construction below, which also omits the NC.
      if constexpr (D == Dim::_1D) {
        return scratch_t::shmem_size(TE);
      } else if constexpr (D == Dim::_2D) {
        return scratch_t::shmem_size(TE, TE);
      } else {
        return scratch_t::shmem_size(TE, TE, TE);
      }
    }

    Inline void operator()(const member_t& team) const {
      const auto tile_id = static_cast<ncells_t>(team.league_rank());
      /**
       * Tile coordinates (tile-grid indices) → tile origin in **active**
       * cell coords (no ghost offset). Using ncells_t to match the
       * linearised tile index produced by SortSpatially.
       */
      ncells_t tx1 = 0, tx2 = 0, tx3 = 0;
      if constexpr (D == Dim::_1D) {
        tx1 = tile_id;
      } else if constexpr (D == Dim::_2D) {
        tx1 = tile_id / ntx2;
        tx2 = tile_id - tx1 * ntx2;
      } else {
        const auto plane = ntx2 * ntx3;
        tx1              = tile_id / plane;
        const auto rem   = tile_id - tx1 * plane;
        tx2              = rem / ntx3;
        tx3              = rem - tx2 * ntx3;
      }
      /**
       * origin_active = lowest active-cell index in the tile (no ghost).
       * origin_F      = same value translated into F's storage coordinate
       *                 (i.e. plus N_GHOSTS).
       * origin_F_low  = F coordinate of scratch index 0 (i.e. origin_F - HALO).
       * local index `li` in scratch ↔ global F index `gi = li + origin_F_low`.
       */
      const int origin_F1_low = static_cast<int>(tx1 * T_TILE) +
                                static_cast<int>(N_GHOSTS) - HALO;
      const int origin_F2_low = static_cast<int>(tx2 * T_TILE) +
                                static_cast<int>(N_GHOSTS) - HALO;
      const int origin_F3_low = static_cast<int>(tx3 * T_TILE) +
                                static_cast<int>(N_GHOSTS) - HALO;

      // Clamp the tile's particle slice to the live array: slots past
      // `npart` may hold stale (possibly alive-tagged) data from a prior
      // step's compaction and must not be re-deposited.
      const auto t_lo    = tile_offsets(tile_id);
      const auto t_hi    = tile_offsets(tile_id + 1u);
      const auto p_begin = (t_lo < npart) ? t_lo : npart;
      const auto p_end   = (t_hi < npart) ? t_hi : npart;

      // Allocate scratch, cooperatively zero-fill it, run the body over
      // the tile's particles (one thread per particle, each through its
      // own Sink), then cooperatively flush scratch to the global field
      // with a bounds clip against the storage extent.
      if constexpr (D == Dim::_1D) {
        scratch_t scr { team.team_scratch(0), TE };
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TE * NCi),
                             [&](const int idx) {
                               const int li = idx / NCi;
                               const int c  = idx - li * NCi;
                               scr(li, c)   = ZERO;
                             });
        team.team_barrier();

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, p_begin, p_end),
                             [&](prtlidx_t p) {
                               Sink sink { scr,           F,
                                           origin_F1_low, 0,
                                           0,             ext1,
                                           0,             0,
                                           comp_offset };
                               body(p, sink);
                             });
        team.team_barrier();

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TE * NCi),
                             [&](const int idx) {
                               const int li = idx / NCi;
                               const int c  = idx - li * NCi;
                               const int gi = li + origin_F1_low;
                               if (gi < 0 or gi >= ext1) {
                                 return;
                               }
                               const real_t v = scr(li, c);
                               if (v != ZERO) {
                                 Kokkos::atomic_add(&F(gi, comp_offset + c), v);
                               }
                             });
      } else if constexpr (D == Dim::_2D) {
        scratch_t scr { team.team_scratch(0), TE, TE };
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, SQR(TE) * NCi),
                             [&](const int idx) {
                               const int lij  = idx / NCi;
                               const int c    = idx - lij * NCi;
                               const int li   = lij / TE;
                               const int lj   = lij - li * TE;
                               scr(li, lj, c) = ZERO;
                             });
        team.team_barrier();

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, p_begin, p_end),
                             [&](prtlidx_t p) {
                               Sink sink { scr,           F,
                                           origin_F1_low, origin_F2_low,
                                           0,             ext1,
                                           ext2,          0,
                                           comp_offset };
                               body(p, sink);
                             });
        team.team_barrier();

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, SQR(TE) * NCi),
                             [&](const int idx) {
                               const int lij = idx / NCi;
                               const int c   = idx - lij * NCi;
                               const int li  = lij / TE;
                               const int lj  = lij - li * TE;
                               const int gi  = li + origin_F1_low;
                               const int gj  = lj + origin_F2_low;
                               if ((gi < 0 or gi >= ext1) or
                                   (gj < 0 or gj >= ext2)) {
                                 return;
                               }
                               const real_t v = scr(li, lj, c);
                               if (v != ZERO) {
                                 Kokkos::atomic_add(&F(gi, gj, comp_offset + c),
                                                    v);
                               }
                             });
      } else if constexpr (D == Dim::_3D) {
        scratch_t scr { team.team_scratch(0), TE, TE, TE };
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, CUBE(TE) * NCi),
                             [&](const int idx) {
                               const int lijk     = idx / NCi;
                               const int c        = idx - lijk * NCi;
                               const int li       = lijk / (TE * TE);
                               const int rem      = lijk - li * TE * TE;
                               const int lj       = rem / TE;
                               const int lk       = rem - lj * TE;
                               scr(li, lj, lk, c) = ZERO;
                             });
        team.team_barrier();

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, p_begin, p_end),
                             [&](prtlidx_t p) {
                               Sink sink { scr,           F,
                                           origin_F1_low, origin_F2_low,
                                           origin_F3_low, ext1,
                                           ext2,          ext3,
                                           comp_offset };
                               body(p, sink);
                             });
        team.team_barrier();

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, CUBE(TE) * NCi),
                             [&](const int idx) {
                               const int lijk = idx / NCi;
                               const int c    = idx - lijk * NCi;
                               const int li   = lijk / (TE * TE);
                               const int rem  = lijk - li * TE * TE;
                               const int lj   = rem / TE;
                               const int lk   = rem - lj * TE;
                               const int gi   = li + origin_F1_low;
                               const int gj   = lj + origin_F2_low;
                               const int gk   = lk + origin_F3_low;
                               if ((gi < 0 or gi >= ext1) or
                                   (gj < 0 or gj >= ext2) or
                                   (gk < 0 or gk >= ext3)) {
                                 return;
                               }
                               const real_t v = scr(li, lj, lk, c);
                               if (v != ZERO) {
                                 Kokkos::atomic_add(
                                   &F(gi, gj, gk, comp_offset + c),
                                   v);
                               }
                             });
      }
    }
  };

  /**
   * @brief Build the TeamPolicy for a tiled-scatter kernel (host).
   *
   * Factors the policy boilerplate shared by every tiled launcher:
   * `TeamPolicy(ntiles, AUTO)` + per-team scratch from
   * `Kern::scratch_bytes()`. The default (team_size_req == 0) leaves
   * Kokkos::AUTO, which sizes the team from the backend occupancy
   * heuristic. A positive request (runtime param
   * `algorithms.deposit.team_policy_team_size`) overrides it, clamped to
   * the scratch/backend-feasible maximum so an over-large request cannot
   * abort the launch (Kokkos errors when team_size > team_size_max). No
   * portable subgroup rounding is applied; pick a multiple of the device
   * subgroup width (printed per arch by ideal_tile_size.py) for the best
   * occupancy.
   *
   * The *tail pass* over `[npart_partitioned, npart)` and the *flat
   * fallback* (empty tile layout) stay at each call site — they need the
   * engine-specific flat kernel.
   */
  template <class Kern>
  inline auto MakeTiledPolicy(const Kern& kern,
                              ncells_t    ntiles,
                              int         team_size_req) -> Kokkos::TeamPolicy<> {
    const auto           scratch = Kokkos::PerTeam(Kern::scratch_bytes());
    Kokkos::TeamPolicy<> policy(static_cast<int>(ntiles), Kokkos::AUTO);
    policy.set_scratch_size(0, scratch);
    if (team_size_req > 0) {
      const int ts_max = policy.team_size_max(kern, Kokkos::ParallelForTag {});
      int       ts     = team_size_req;
      if (ts > ts_max) {
        raise::Warning(
          fmt::format("algorithms.deposit.team_policy_team_size = %d exceeds "
                      "the tiled-scatter maximum %d on this backend; clamping "
                      "to %d",
                      team_size_req,
                      ts_max,
                      ts_max),
          HERE);
        ts = ts_max;
      }
      policy = Kokkos::TeamPolicy<>(static_cast<int>(ntiles), ts);
      policy.set_scratch_size(0, scratch);
      logger::Checkpoint(fmt::format("Tiled scatter: explicit team size %d", ts),
                         HERE);
    }
    return policy;
  }

} // namespace kernel

#endif // TEAM_POLICY

#endif // KERNELS_TILED_SCATTER_HPP
