/**
 * @file kernels/deposition/currents/tiled.hpp
 * @brief Tiled current deposition kernel with per-team SLM scratch.
 *
 * @note Team-policy (one team per spatial tile, accumulates into team SLM scratch with
 *     atomic adds, then flushes to global J). Available when `tiled_deposit=ON`
 *     (`#if defined(TILED_DEPOSIT)`). A thin wrapper over the generic
 *     `kernel::TiledScatter_kernel` harness (kernels/tiled_scatter.hpp):
 *     the currents-specific part is only `CurrentsDepositBody`, which
 *     computes the per-particle footprint from the stored i/i_prev and
 *     feeds `DepositOneParticle` through the harness sink.
 *
 * @implements
 *   - kernel::CurrentsDepositBody<>            (TILED_DEPOSIT only)
 *   - kernel::DepositCurrentsTiled_kernel<>    (TILED_DEPOSIT only)
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_DEPOSITION_CURRENTS_TILED_HPP
#define KERNELS_DEPOSITION_CURRENTS_TILED_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "kernels/deposition/currents/single-particle.hpp"
#include "kernels/tiled_scatter.hpp"

#include <Kokkos_Core.hpp>

namespace kernel {
  using namespace ntt;

#if defined(TILED_DEPOSIT)
  /**
   * @brief Per-particle body of the tiled current deposit.
   *
   * Implements the `TiledScatter_kernel` body contract: computes the
   * particle's conservative deposit footprint from the stored `i`/`i_prev`
   * pair, `select()`s it on the sink (which decides SLM scratch vs the
   * per-particle global escape valve), then runs the shared
   * `DepositOneParticle` math with the sink as the `deposit_at` callback.
   *
   * One-sided footprint reach: the deposit writes at most FOOTPRINT_REACH
   * cells above max(i,i_prev) (and fewer below min), so
   * [min(i,i_prev) - FOOTPRINT_REACH, max(i,i_prev) + FOOTPRINT_REACH] in
   * cell coords conservatively bounds every deposited cell for any order
   * (Esirkepov reaches max+O; O=0 zigzag reaches max+1). When the whole
   * footprint fits the tile scratch window, every deposited cell is
   * provably inside it and the scratch writes need no per-cell bounds
   * test; otherwise the WHOLE particle goes to the bounds-clipped global
   * path (see tiled_scatter.hpp for why this is charge-conserving).
   */
  template <SimEngine::type S, MetricClass M, unsigned short O>
  struct CurrentsDepositBody {
    static_assert(O <= 11u, "Shape order O must be <= 11");
    static constexpr int FOOTPRINT_REACH = (O == 0u) ? 1 : static_cast<int>(O);

    ParticleArrays prtls;
    const M        metric;
    const real_t   charge, inv_dt;

    CurrentsDepositBody(const ParticleArrays& prtls,
                        const M&              metric,
                        real_t                charge,
                        real_t                dt)
      : prtls { prtls }
      , metric { metric }
      , charge { charge }
      , inv_dt { ONE / dt } {}

    template <class Sink>
    Inline void operator()(prtlidx_t p, Sink& sink) const {
      constexpr auto D = M::Dim;
      const int      G = static_cast<int>(N_GHOSTS);
      const int      i1c = prtls.i1(p), i1p = prtls.i1_prev(p);
      if constexpr (D == Dim::_1D) {
        sink.select((i1c < i1p ? i1c : i1p) + G - FOOTPRINT_REACH,
                    (i1c > i1p ? i1c : i1p) + G + FOOTPRINT_REACH);
      } else if constexpr (D == Dim::_2D) {
        const int i2c = prtls.i2(p), i2p = prtls.i2_prev(p);
        sink.select((i1c < i1p ? i1c : i1p) + G - FOOTPRINT_REACH,
                    (i1c > i1p ? i1c : i1p) + G + FOOTPRINT_REACH,
                    (i2c < i2p ? i2c : i2p) + G - FOOTPRINT_REACH,
                    (i2c > i2p ? i2c : i2p) + G + FOOTPRINT_REACH);
      } else {
        const int i2c = prtls.i2(p), i2p = prtls.i2_prev(p);
        const int i3c = prtls.i3(p), i3p = prtls.i3_prev(p);
        sink.select((i1c < i1p ? i1c : i1p) + G - FOOTPRINT_REACH,
                    (i1c > i1p ? i1c : i1p) + G + FOOTPRINT_REACH,
                    (i2c < i2p ? i2c : i2p) + G - FOOTPRINT_REACH,
                    (i2c > i2p ? i2c : i2p) + G + FOOTPRINT_REACH,
                    (i3c < i3p ? i3c : i3p) + G - FOOTPRINT_REACH,
                    (i3c > i3p ? i3c : i3p) + G + FOOTPRINT_REACH);
      }
      DepositOneParticle<S, M, O>(p, prtls, metric, charge, inv_dt, sink);
    }
  };

  /**
   * @brief Tiled current-deposition kernel.
   *
   * A thin wrapper: `TiledScatter_kernel` (kernels/tiled_scatter.hpp)
   * carries the whole team/scratch/flush harness — one team per spatial
   * tile, per-team scratch of shape `(T_TILE + 2*HALO)^D x 3`, SLM atomics
   * for in-tile particles, the per-particle global escape valve, the
   * particle-slice clamp to the live `npart`, and the bounds-clipped
   * cooperative flush to global J. This class only fixes the template
   * arguments (NC = NG = 3, REACH = STENCIL_REACH(O)) and keeps the
   * public name + constructor signature the engine launchers use.
   *
   * Supports `O in {0, ..., 11}`. `O == 0` (zigzag) is wired for
   * A/B benchmarking against the flat scatter-view kernel — its narrow
   * stencil typically makes scratch alloc/zero/flush overhead a
   * regression there, but it's good to be able to measure the
   * crossover. To revert and use flat for zigzag-only builds, change
   * the dispatch in `engines/srpic/currents.h` from
   * `#if defined(TILED_DEPOSIT)` to
   * `#if defined(TILED_DEPOSIT) && (SHAPE_ORDER > 0)`.
   *
   * Particle iteration order is governed by `tile_offsets`: tile `t`
   * owns particles `[tile_offsets(t), tile_offsets(t+1))`, post-sort.
   * `SortSpatially` (`particles_sort.cpp`) is responsible for keeping
   * the SoA arrays consistent with that. Particles appended past the
   * partition are deposited by the launcher's flat tail pass (see the
   * partition-coverage note in tiled_scatter.hpp).
   *
   * **Halo sizing.** Sort runs at the end of a step (see `srpic.hpp`); a
   * particle is pushed once per step thereafter, so its `min(i, i_prev)`
   * may differ from the bin key by one cell of drift per step elapsed
   * since the last sort. The scratch HALO is `STENCIL_REACH(O) + DRIFT`:
   *
   *   stencil_reach(O) — maximum cells the deposit writes ABOVE
   *   min(i, i_prev) under CFL |v * dt/dx| <= 1/2:
   *   - O == 0 (zigzag):  writes { i_prev, i_prev+1, i, i+1 } => +2
   *   - O >= 1 Esirkepov: `for_deposit` returns an (O+2)-wide
   *     array but only O+1 entries are non-zero, and the union
   *     window satisfies `i_max - i_min <= O+1` (see
   *     particle_shapes.hpp::for_deposit). The genuine one-sided
   *     reach above min(i, i_prev) is therefore O, not O+1.
   *
   * `DRIFT` (the `tiled_deposit_drift` CMake knob) and the escape-valve /
   * charge-conservation argument are documented on the harness.
   */
  template <SimEngine::type S, MetricClass M, unsigned short O, unsigned short T_TILE>
  class DepositCurrentsTiled_kernel
    : public TiledScatter_kernel<M::Dim,
                                 3,
                                 3,
                                 ((O == 0u) ? 2 : static_cast<int>(O)),
                                 T_TILE,
                                 CurrentsDepositBody<S, M, O>> {
    static_assert(O <= 11u, "Shape order O must be <= 11");

    using body_t = CurrentsDepositBody<S, M, O>;
    using base_t = TiledScatter_kernel<M::Dim,
                                       3,
                                       3,
                                       ((O == 0u) ? 2 : static_cast<int>(O)),
                                       T_TILE,
                                       body_t>;

  public:
    DepositCurrentsTiled_kernel(const ndfield_t<M::Dim, 3>& cur,
                                const ParticleArrays&       prtls,
                                const M&                    metric,
                                real_t                      charge,
                                real_t                      dt,
                                const TileLayout<M::Dim>&   layout,
                                npart_t                     npart)
      : base_t { cur, body_t { prtls, metric, charge, dt }, layout, npart } {}
  };
#endif // TILED_DEPOSIT

} // namespace kernel

#endif // KERNELS_DEPOSITION_CURRENTS_TILED_HPP
