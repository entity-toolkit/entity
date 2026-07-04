/**
 * @file engines/srpic/currents.h
 * @brief Current deposition and filtering routines for the SRPIC engine
 * @implements
 *   - ntt::srpic::CallDepositKernel<> -> void                 (flat path)
 *   - ntt::srpic::CallDepositKernelTiled<> -> void            (TEAM_POLICY)
 *   - ntt::srpic::CurrentsDeposit<> -> void
 *   - ntt::srpic::CurrentsFilter<> -> void
 * @namespaces:
 *   - ntt::srpic::
 */

#ifndef ENGINES_SRPIC_CURRENTS_H
#define ENGINES_SRPIC_CURRENTS_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/log.h"
#include "utils/param_container.h"

#include "engines/srpic/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "kernels/currents_deposit.hpp"
#include "kernels/digital_filter.hpp"

#include <utility>

namespace ntt {
  namespace srpic {

    template <SRMetricClass M, unsigned short O>
    void CallDepositKernel(const Particles<M::Dim, M::CoordType>& species,
                           const M&                               local_metric,
                           const scatter_ndfield_t<M::Dim, 3>&    scatter_cur,
                           real_t                                 dt) {
      Kokkos::parallel_for("CurrentsDeposit",
                           species.rangeActiveParticles(),
                           kernel::DepositCurrents_kernel<SimEngine::SRPIC, M, O>(
                             scatter_cur,
                             species,
                             local_metric,
                             (real_t)(species.charge()),
                             dt));
    }

#if defined(TEAM_POLICY)
    /**
     * @brief Tiled deposit launcher (TeamPolicy + per-team scratch).
     *
     * Iterates over `tile_layout.ntiles_total` teams; each team accumulates
     * its tile's particle contributions in SLM scratch and atomically
     * flushes to the global J. Requires the species to have been sorted
     * with `team_policy` enabled (`tile_layout` populated by
     * `SortSpatially`).
     *
     * Falls back to the flat kernel if `tile_offsets` is empty — this
     * happens on the first step before the first sort, or for very small
     * species that exited early in `SortSpatially`. The fallback uses the
     * passed-in `scatter_cur` so the caller still composes correctly.
     */
    template <SRMetricClass M, unsigned short O>
    void CallDepositKernelTiled(const Particles<M::Dim, M::CoordType>& species,
                                const M&                    local_metric,
                                const ndfield_t<M::Dim, 3>& cur,
                                real_t                      dt,
                                int                         team_size_req) {
      static_assert(O <= 11u, "Shape order must be <= 11");
      constexpr unsigned short T = static_cast<unsigned short>(
        TEAM_POLICY_TILE_SIZE);
      const auto& layout = species.tile_layout();
      raise::ErrorIf(layout.ntiles_total == 0u,
                     "CallDepositKernelTiled: tile_layout has 0 tiles — call "
                     "SortSpatially before CurrentsDeposit",
                     HERE);
      raise::ErrorIf(layout.tile_offsets.extent(0) != layout.ntiles_total + 1u,
                     "CallDepositKernelTiled: tile_offsets size inconsistent "
                     "with ntiles_total",
                     HERE);

      auto deposit_kernel =
        kernel::DepositCurrentsTiled_kernel<SimEngine::SRPIC, M, O, T> {
          cur,    species, local_metric, (real_t)(species.charge()),
          dt,     layout,  species.npart()
        };

      const auto scratch = Kokkos::PerTeam(
        decltype(deposit_kernel)::scratch_bytes());

      // Team (work-group) size. The default (team_size_req == 0) leaves
      // Kokkos::AUTO, which sizes the team from the backend occupancy
      // heuristic. A positive `algorithms.deposit.team_policy_team_size`
      // overrides it, clamped to the scratch/backend-feasible maximum so an
      // over-large request cannot abort the launch (Kokkos errors when
      // team_size > team_size_max). No portable subgroup rounding is applied;
      // pick a multiple of the device subgroup width (printed per arch by
      // ideal_tile_size.py) for the best occupancy.
      Kokkos::TeamPolicy<> policy(static_cast<int>(layout.ntiles_total),
                                  Kokkos::AUTO);
      policy.set_scratch_size(0, scratch);
      if (team_size_req > 0) {
        const int ts_max = policy.team_size_max(deposit_kernel,
                                                Kokkos::ParallelForTag {});
        int       ts     = team_size_req;
        if (ts > ts_max) {
          raise::Warning(
            fmt::format("algorithms.deposit.team_policy_team_size = %d exceeds "
                        "the tiled-deposit maximum %d on this backend; clamping "
                        "to %d",
                        team_size_req,
                        ts_max,
                        ts_max),
            HERE);
          ts = ts_max;
        }
        policy = Kokkos::TeamPolicy<>(static_cast<int>(layout.ntiles_total), ts);
        policy.set_scratch_size(0, scratch);
        logger::Checkpoint(
          fmt::format("Tiled deposit: explicit team size %d", ts),
          HERE);
      }
      Kokkos::parallel_for("CurrentsDepositTiled", policy, deposit_kernel);

      // Particles appended since the last sort (injection / MPI receive on a
      // no-sort step) live past the partition and are not visited by any team
      // above. Deposit that tail [npart_partitioned, npart) with the flat
      // scatter-view kernel so every active particle is deposited exactly
      // once. The range is empty when the species was just sorted (the
      // every-step-sorted common case), so this is a no-op there.
      if (species.npart() > layout.npart_partitioned) {
        // `cur` is a const ref; take a non-const View handle (shallow copy,
        // shares storage) so the scatter view can contribute back into it.
        auto cur_nc      = cur;
        auto scatter_cur = Kokkos::Experimental::create_scatter_view(cur_nc);
        Kokkos::parallel_for(
          "CurrentsDepositTiledTail",
          CreateParticleRangePolicy<Dim::_1D>({ layout.npart_partitioned },
                                              { species.npart() }),
          kernel::DepositCurrents_kernel<SimEngine::SRPIC, M, O>(
            scatter_cur,
            species,
            local_metric,
            (real_t)(species.charge()),
            dt));
        Kokkos::Experimental::contribute(cur_nc, scatter_cur);
      }
    }
#endif // TEAM_POLICY

    template <SRMetricClass M>
    void CurrentsDeposit(Domain<SimEngine::SRPIC, M>& domain,
                         const prm::Parameters&       engine_params) {
      const auto dt = engine_params.get<real_t>("dt");
      Kokkos::deep_copy(domain.fields.cur, ZERO);

#if defined(TEAM_POLICY)
      // Optional runtime override for the tiled-deposit team (work-group) size;
      // 0 (default) keeps Kokkos::AUTO. Clamped to the backend max in the
      // launcher (see CallDepositKernelTiled).
      const auto team_size_req = static_cast<int>(
        engine_params.get<std::size_t>("team_policy_team_size",
                                       std::optional<std::size_t> { 0u }));

      // Tiled deposit. Correctness no longer depends on the SoA being in a
      // "sorted" state at deposit time — the tiled kernel handles a stale
      // partition per-particle:
      //   - a particle whose full stencil has drifted out of its tile is
      //     deposited straight to the global J view (the per-particle escape
      //     valve); `team_policy_drift` sizes the scratch halo so the
      //     common in-tile case stays in fast SLM (see currents_deposit.hpp);
      //   - particles dead-tagged in place since the sort are clamped out by
      //     the kernel and skipped by the dead-tag test;
      //   - particles appended past the partition since the sort (injection /
      //     MPI receive on a no-sort step) are deposited by the launcher's
      //     flat tail pass over [npart_partitioned, npart).
      // Together these cover every active particle exactly once for any sort
      // interval. The only case the tiled kernel cannot serve is the very
      // first step, before any SortSpatially has populated a layout; that
      // species takes the flat scatter-view path for that step alone.
      for (auto& species : domain.species) {
        if ((species.pusher() == ParticlePusher::NONE) or
            (species.npart() == 0) or cmp::AlmostZero_host(species.charge())) {
          continue;
        }
        const auto& layout = species.tile_layout();
        if (layout.ntiles_total == 0u or layout.tile_offsets.extent(0) == 0u) {
          logger::Checkpoint(
            fmt::format("Launching currents deposit (flat, no sort yet) for "
                        "%d [%s] : %lu %f",
                        species.index(),
                        species.label().c_str(),
                        species.npart(),
                        (double)species.charge()),
            HERE);
          auto scatter_cur = Kokkos::Experimental::create_scatter_view(
            domain.fields.cur);
          CallDepositKernel<M, SHAPE_ORDER>(species,
                                            domain.mesh.metric,
                                            scatter_cur,
                                            dt);
          Kokkos::Experimental::contribute(domain.fields.cur, scatter_cur);
        } else {
          logger::Checkpoint(
            fmt::format("Launching tiled currents deposit for %d [%s] : %lu %f",
                        species.index(),
                        species.label().c_str(),
                        species.npart(),
                        (double)species.charge()),
            HERE);
          CallDepositKernelTiled<M, SHAPE_ORDER>(species,
                                                 domain.mesh.metric,
                                                 domain.fields.cur,
                                                 dt,
                                                 team_size_req);
        }
      }
#else
      auto scatter_cur = Kokkos::Experimental::create_scatter_view(
        domain.fields.cur);
      for (auto& species : domain.species) {
        if ((species.pusher() == ParticlePusher::NONE) or
            (species.npart() == 0) or cmp::AlmostZero_host(species.charge())) {
          continue;
        }
        logger::Checkpoint(
          fmt::format("Launching currents deposit kernel for %d [%s] : %lu %f",
                      species.index(),
                      species.label().c_str(),
                      species.npart(),
                      (double)species.charge()),
          HERE);

        CallDepositKernel<M, SHAPE_ORDER>(species, domain.mesh.metric, scatter_cur, dt);
      }
      Kokkos::Experimental::contribute(domain.fields.cur, scatter_cur);
#endif
    }

    template <SRMetricClass M>
    void CurrentsFilter(Metadomain<SimEngine::SRPIC, M>& metadomain,
                        Domain<SimEngine::SRPIC, M>&     domain,
                        const SimulationParams&          params) {
      logger::Checkpoint("Launching currents filtering kernels", HERE);
      const auto nfilter = params.template get<unsigned short>(
        "algorithms.current_filters");
      if (nfilter == 0u) {
        return;
      }
      tuple_t<ncells_t, M::Dim> size;
      if constexpr (M::Dim == Dim::_1D || M::Dim == Dim::_2D || M::Dim == Dim::_3D) {
        size[0] = domain.mesh.n_active(in::x1);
      }
      if constexpr (M::Dim == Dim::_2D || M::Dim == Dim::_3D) {
        size[1] = domain.mesh.n_active(in::x2);
      }
      if constexpr (M::Dim == Dim::_3D) {
        size[2] = domain.mesh.n_active(in::x3);
      }

      // The filter ping-pongs `cur` <-> scratch `buff`: one up-front copy
      // seeds `buff` with valid ghost cells (the kernel writes only the
      // cells in its launch range), then each pass filters the input into
      // the other buffer and swaps the View handles so `cur` again names the
      // result. `buff` is pure scratch (also reused by CommunicateFields), so
      // the permanent handle swap is transparent and `cur` always names the
      // result. The single seeding copy preserves physical-boundary ghosts
      // (conductor/match/atmosphere/...), which neither the kernel nor the
      // MPI exchange refresh.
      Kokkos::deep_copy(domain.fields.buff, domain.fields.cur);

      const auto flds_bc = domain.mesh.flds_bc();

      // Reduced-exchange ghost-margin scheme (all coordinate types). One halo
      // exchange refreshes N_GHOSTS ghost layers — enough for N_GHOSTS passes
      // of the 3-point binomial if each pass also recomputes the inner ghost
      // layers it will need next. We therefore extend the launch range by a
      // shrinking margin `m` into the ghost zone, but only on comm-refreshed
      // sides (PERIODIC self-wrap or SYNC inter-domain), where the ghost cell
      // is interior physics. Physical-boundary ghosts are never written or
      // refreshed, exactly as in the per-pass loop, so the result is identical
      // for every BC — while doing one exchange per N_GHOSTS passes instead of
      // one per pass. (Entering the loop the ghosts are valid to distance
      // N_GHOSTS: srpic.hpp runs CommunicateFields(J) immediately before
      // CurrentsFilter.)
      //
      // Non-Cartesian axis: the theta (x2) direction is self-contained in the
      // filter kernel — the axis branches only ever read/write within
      // [i2_min, i2_max] and never cross the axis — so the axis needs no halo
      // exchange at all (the axis current fold is done once by
      // SynchronizeFields(J) before this function). The single coordinate
      // dependency is that the axis cell sits at i_max(x2), one past the active
      // range, and must be filtered on every pass. We therefore add a fixed +1
      // to the x2 upper bound when that side is AXIS — a physical boundary, so
      // never the shrinking comm margin. This folds the old RangeWithAxisBCs
      // fixup into make_range, letting the same loop serve every CoordType.
      const int  G = static_cast<int>(N_GHOSTS);
      const auto comm_side = [](FldsBC b) {
        return (b == FldsBC::PERIODIC) or (b == FldsBC::SYNC);
      };
      bool ext_lo[3] = { false, false, false };
      bool ext_hi[3] = { false, false, false };
      for (auto d { 0 }; d < static_cast<int>(M::Dim); ++d) {
        ext_lo[d] = comm_side(flds_bc[d].first);
        ext_hi[d] = comm_side(flds_bc[d].second);
      }
      // AXIS at the upper x2 boundary needs the axis cell (i_max(x2)) included
      // every pass; matches srpic::RangeWithAxisBCs. (The lower-x2 axis cell is
      // already i_min(x2), so no fixup is needed there.)
      bool axis_hi_x2 = false;
      if constexpr (M::CoordType != Coord::Cartesian and
                    (M::Dim == Dim::_2D or M::Dim == Dim::_3D)) {
        axis_hi_x2 = (flds_bc[1].second == FldsBC::AXIS);
      }
      const auto make_range = [&](int m) -> range_t<M::Dim> {
        const auto ml = [&](int d) -> ncells_t {
          return ext_lo[d] ? static_cast<ncells_t>(m) : 0u;
        };
        const auto mh = [&](int d) -> ncells_t {
          if (ext_hi[d]) {
            return static_cast<ncells_t>(m);
          }
          // axis cell fixup (x2 == dimension index 1); mutually exclusive with
          // the comm margin since AXIS is not a comm side
          if (d == 1 and axis_hi_x2) {
            return 1u;
          }
          return 0u;
        };
        if constexpr (M::Dim == Dim::_1D) {
          return CreateRangePolicy<Dim::_1D>(
            { domain.mesh.i_min(in::x1) - ml(0) },
            { domain.mesh.i_max(in::x1) + mh(0) });
        } else if constexpr (M::Dim == Dim::_2D) {
          return CreateRangePolicy<Dim::_2D>(
            { domain.mesh.i_min(in::x1) - ml(0),
              domain.mesh.i_min(in::x2) - ml(1) },
            { domain.mesh.i_max(in::x1) + mh(0),
              domain.mesh.i_max(in::x2) + mh(1) });
        } else {
          return CreateRangePolicy<Dim::_3D>(
            { domain.mesh.i_min(in::x1) - ml(0),
              domain.mesh.i_min(in::x2) - ml(1),
              domain.mesh.i_min(in::x3) - ml(2) },
            { domain.mesh.i_max(in::x1) + mh(0),
              domain.mesh.i_max(in::x2) + mh(1),
              domain.mesh.i_max(in::x3) + mh(2) });
        }
      };
      int m = G - 1;
      for (auto i { 0u }; i < nfilter; ++i) {
        Kokkos::parallel_for(
          "CurrentsFilter",
          make_range(m),
          kernel::DigitalFilter_kernel<M::Dim, M::CoordType>(
            domain.fields.buff,
            domain.fields.cur,
            size,
            flds_bc));
        std::swap(domain.fields.cur, domain.fields.buff);
        --m;
        if (m < 0 or i == nfilter - 1u) {
          // refresh ghosts to distance G (and leave them valid for the
          // downstream field solver after the final pass)
          metadomain.CommunicateFields(domain, Comm::J);
          m = G - 1;
        }
      }
    }

  } // namespace srpic
} // namespace ntt

#endif // ENGINES_SRPIC_CURRENTS_H
