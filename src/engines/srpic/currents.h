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
                             species.i1,
                             species.i2,
                             species.i3,
                             species.i1_prev,
                             species.i2_prev,
                             species.i3_prev,
                             species.dx1,
                             species.dx2,
                             species.dx3,
                             species.dx1_prev,
                             species.dx2_prev,
                             species.dx3_prev,
                             species.ux1,
                             species.ux2,
                             species.ux3,
                             species.phi,
                             species.weight,
                             species.tag,
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
    void CallDepositKernelTiled(
      const Particles<M::Dim, M::CoordType>& species,
      const M&                               local_metric,
      const ndfield_t<M::Dim, 3>&            cur,
      real_t                                 dt) {
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

      using kernel_t = kernel::DepositCurrents_kernel_tiled<SimEngine::SRPIC,
                                                            M,
                                                            O,
                                                            T>;
      kernel_t kern { cur,
                      species.i1,
                      species.i2,
                      species.i3,
                      species.i1_prev,
                      species.i2_prev,
                      species.i3_prev,
                      species.dx1,
                      species.dx2,
                      species.dx3,
                      species.dx1_prev,
                      species.dx2_prev,
                      species.dx3_prev,
                      species.ux1,
                      species.ux2,
                      species.ux3,
                      species.phi,
                      species.weight,
                      species.tag,
                      local_metric,
                      (real_t)(species.charge()),
                      dt,
                      layout };

      Kokkos::TeamPolicy<> policy(static_cast<int>(layout.ntiles_total),
                                  Kokkos::AUTO);
      policy.set_scratch_size(0, Kokkos::PerTeam(kernel_t::scratch_bytes()));
      Kokkos::parallel_for("CurrentsDepositTiled", policy, kern);
    }
#endif // TEAM_POLICY

    template <SRMetricClass M>
    void CurrentsDeposit(Domain<SimEngine::SRPIC, M>& domain,
                         const prm::Parameters&       engine_params) {
      const auto dt = engine_params.get<real_t>("dt");
      Kokkos::deep_copy(domain.fields.cur, ZERO);

#if defined(TEAM_POLICY)

      // First-step fallback: if any contributing species has not been
      // sorted yet (tile_layout still empty), fall back to the flat
      // scatter-view path for that step. Subsequent steps see populated
      // layouts and use the tiled kernel.
      bool any_unsorted = false;
      for (auto& species : domain.species) {
        if ((species.pusher() == ParticlePusher::NONE) or
            (species.npart() == 0) or cmp::AlmostZero_host(species.charge())) {
          continue;
        }
        if (species.tile_layout().ntiles_total == 0u or
            species.tile_layout().tile_offsets.extent(0) == 0u) {
          any_unsorted = true;
          break;
        }
      }
      if (any_unsorted) {
        auto scatter_cur = Kokkos::Experimental::create_scatter_view(
          domain.fields.cur);
        for (auto& species : domain.species) {
          if ((species.pusher() == ParticlePusher::NONE) or
              (species.npart() == 0) or cmp::AlmostZero_host(species.charge())) {
            continue;
          }
          logger::Checkpoint(
            fmt::format("Launching currents deposit (flat fallback, no sort yet) "
                        "for %d [%s] : %lu %f",
                        species.index(),
                        species.label().c_str(),
                        species.npart(),
                        (double)species.charge()),
            HERE);
          CallDepositKernel<M, SHAPE_ORDER>(species,
                                             domain.mesh.metric,
                                             scatter_cur,
                                             dt);
        }
        Kokkos::Experimental::contribute(domain.fields.cur, scatter_cur);
      } else {
        for (auto& species : domain.species) {
          if ((species.pusher() == ParticlePusher::NONE) or
              (species.npart() == 0) or cmp::AlmostZero_host(species.charge())) {
            continue;
          }
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
