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
                                real_t                      dt) {
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

      Kokkos::TeamPolicy<> policy(static_cast<int>(layout.ntiles_total),
                                  Kokkos::AUTO);
      policy.set_scratch_size(
        0,
        Kokkos::PerTeam(decltype(deposit_kernel)::scratch_bytes()));
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

    template <SRMetricClass M>
    void CurrentsDeposit(Domain<SimEngine::SRPIC, M>& domain,
                         const prm::Parameters&       engine_params) {
      const auto dt = engine_params.get<real_t>("dt");
      Kokkos::deep_copy(domain.fields.cur, ZERO);

#if defined(TEAM_POLICY)

      // Tiled deposit. Correctness no longer depends on the SoA being in a
      // "sorted" state at deposit time — the tiled kernel handles a stale
      // partition per-particle:
      //   - a particle whose full stencil has drifted out of its tile is
      //     deposited straight to the global J view (the per-particle escape
      //     valve); `team_policy_sort_interval` sizes the scratch halo so the
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
                                                 dt);
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
      auto       range   = srpic::RangeWithAxisBCs(domain);
      const auto nfilter = params.template get<unsigned short>(
        "algorithms.current_filters");
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
      // !TODO: this needs to be done more efficiently
      for (auto i { 0u }; i < nfilter; ++i) {
        Kokkos::deep_copy(domain.fields.buff, domain.fields.cur);
        Kokkos::parallel_for("CurrentsFilter",
                             range,
                             kernel::DigitalFilter_kernel<M::Dim, M::CoordType>(
                               domain.fields.cur,
                               domain.fields.buff,
                               size,
                               domain.mesh.flds_bc()));
        metadomain.CommunicateFields(domain, Comm::J);
      }
    }

  } // namespace srpic
} // namespace ntt

#endif // ENGINES_SRPIC_CURRENTS_H
