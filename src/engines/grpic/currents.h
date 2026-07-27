/**
 * @file engines/grpic/currents.h
 * @brief Current deposition and filtering routines for the GRPIC engine
 * @implements
 *   - ntt::grpic::CallDepositKernel<> -> void                 (flat path)
 *   - ntt::grpic::CallDepositKernelTiled<> -> void            (TEAM_POLICY)
 *   - ntt::grpic::CurrentsDeposit<> -> void
 *   - ntt::grpic::CurrentsFilter<> -> void
 * @namespaces:
 *   - ntt::grpic::
 */

#ifndef ENGINES_GRPIC_CURRENTS_H
#define ENGINES_GRPIC_CURRENTS_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/log.h"
#include "utils/param_container.h"

#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "kernels/currents_deposit.hpp"
#include "kernels/digital_filter.hpp"

namespace ntt {
  namespace grpic {

    template <GRMetricClass M, unsigned short O>
    void CallDepositKernel(const Particles<M::Dim, M::CoordType>& species,
                           const M&                               local_metric,
                           const scatter_ndfield_t<M::Dim, 3>&    scatter_cur,
                           real_t                                 dt) {
      Kokkos::parallel_for("CurrentsDeposit",
                           species.rangeActiveParticles(),
                           kernel::DepositCurrents_kernel<SimEngine::GRPIC, M, O>(
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
     * Identical in structure to the SRPIC launcher (`engines/srpic/currents.h`):
     * iterates over `tile_layout.ntiles_total` teams; each team accumulates its
     * tile's particle contributions in SLM scratch and atomically flushes to the
     * global J (here `cur0`, the GRPIC half-step current). Requires the species
     * to have been sorted with `team_policy` enabled (`tile_layout` populated by
     * `SortSpatially`).
     *
     * The deposit body (`kernel::DepositOneParticle<SimEngine::GRPIC, M, O>`) is
     * the same shared math used by the flat path — it already carries the GR
     * velocity-recovery branch — so the only engine-specific differences from
     * SRPIC are the `SimEngine::GRPIC` tag and the `cur0` target.
     *
     * Falls back to the flat kernel for the tail `[npart_partitioned, npart)`
     * exactly as SRPIC does; see the per-step coverage note in
     * `kernels/currents_deposit.hpp`.
     */
    template <GRMetricClass M, unsigned short O>
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
        kernel::DepositCurrentsTiled_kernel<SimEngine::GRPIC, M, O, T> {
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
          kernel::DepositCurrents_kernel<SimEngine::GRPIC, M, O>(
            scatter_cur,
            species,
            local_metric,
            (real_t)(species.charge()),
            dt));
        Kokkos::Experimental::contribute(cur_nc, scatter_cur);
      }
    }
#endif // TEAM_POLICY

    template <GRMetricClass M>
    void CurrentsDeposit(Domain<SimEngine::GRPIC, M>& domain,
                         const prm::Parameters&       engine_params) {
      const auto dt = engine_params.get<real_t>("dt");
      // GRPIC deposits the half-step current into `cur0` (the engine no longer
      // pre-zeros it — this is the single source of truth, matching SRPIC).
      Kokkos::deep_copy(domain.fields.cur0, ZERO);

#if defined(TEAM_POLICY)
      // Optional runtime override for the tiled-deposit team (work-group) size;
      // 0 (default) keeps Kokkos::AUTO. Clamped to the backend max in the
      // launcher (see CallDepositKernelTiled).
      const auto team_size_req = static_cast<int>(
        engine_params.get<std::size_t>("team_policy_team_size",
                                       std::optional<std::size_t> { 0u }));

      // Tiled deposit. Correctness no longer depends on the SoA being in a
      // "sorted" state at deposit time — the tiled kernel handles a stale
      // partition per-particle (escape valve for drifted particles, dead-tag
      // clamp, and the launcher's flat tail pass for appended particles). The
      // only case the tiled kernel cannot serve is the very first step, before
      // any SortSpatially has populated a layout; that species takes the flat
      // scatter-view path for that step alone. See engines/srpic/currents.h and
      // kernels/currents_deposit.hpp for the full coverage argument.
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
          auto scatter_cur0 = Kokkos::Experimental::create_scatter_view(
            domain.fields.cur0);
          CallDepositKernel<M, SHAPE_ORDER>(species,
                                            domain.mesh.metric,
                                            scatter_cur0,
                                            dt);
          Kokkos::Experimental::contribute(domain.fields.cur0, scatter_cur0);
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
                                                 domain.fields.cur0,
                                                 dt,
                                                 team_size_req);
        }
      }
#else
      auto scatter_cur0 = Kokkos::Experimental::create_scatter_view(
        domain.fields.cur0);
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

        CallDepositKernel<M, SHAPE_ORDER>(species,
                                          domain.mesh.metric,
                                          scatter_cur0,
                                          dt);
      }
      Kokkos::Experimental::contribute(domain.fields.cur0, scatter_cur0);
#endif
    }

    template <GRMetricClass M>
    void CurrentsFilter(Metadomain<SimEngine::GRPIC, M>& metadomain,
                        Domain<SimEngine::GRPIC, M>&     domain,
                        const SimulationParams&          params) {
      logger::Checkpoint("Launching currents filtering kernels", HERE);
      auto range = CreateRangePolicy<Dim::_2D>(
        { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
        { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
      const auto nfilter = params.template get<unsigned short>(
        "algorithms.current_filters");
      tuple_t<ncells_t, M::Dim> size;
      size[0] = domain.mesh.n_active(in::x1);
      size[1] = domain.mesh.n_active(in::x2);

      // !TODO: this needs to be done more efficiently
      for (unsigned short i = 0; i < nfilter; ++i) {
        Kokkos::deep_copy(domain.fields.buff, domain.fields.cur0);
        Kokkos::parallel_for("CurrentsFilter",
                             range,
                             kernel::DigitalFilter_kernel<M::Dim, M::CoordType>(
                               domain.fields.cur0,
                               domain.fields.buff,
                               size,
                               domain.mesh.flds_bc()));
        metadomain.CommunicateFields(domain, Comm::J); // J0
      }
    }

  } // namespace grpic
} // namespace ntt

#endif // ENGINES_GRPIC_CURRENTS_H
