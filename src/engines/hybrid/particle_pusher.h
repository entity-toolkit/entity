/**
 * @file engines/hybrid/particle_pusher.h
 * @brief Particle push + fused ion-moment deposit drivers for the HYBRID engine
 * @implements
 *   - ntt::hybrid::ParticlePush<>   -> void
 *   - ntt::hybrid::DepositMoments<> -> void
 * @namespaces:
 *   - ntt::hybrid::
 *
 * The Pegasus step (Kunz, Stone & Bai 2014, Fig. 2) performs two ion pushes that
 * BOTH start from the stored state x^(n), v^(n). The pusher kernel deposits the
 * ion moments (N, V) in the SAME pass as the push, so the transient predictor can
 * produce its predicted moments without ever writing the particle arrays — hence
 * no save/restore of x^(n), v^(n) is needed:
 *
 *   DepositMoments(dom, params);           // step 0 only: seed aux with N^(0), V^(0)
 *   ...EMF #1...
 *   ParticlePush(dom, ep, params, false);  // predictor: push (registers) + deposit N', V'
 *   ...EMF #2, Faraday...
 *   ParticlePush(dom, ep, params, true);   // corrector: push + deposit + store x^(n+1)
 *
 * Each call rebuilds the `aux` ScatterView, zeroes aux, runs the kernel (which
 * scatter-deposits), and contributes. The caller then remaps/fills aux ghosts:
 *   SynchronizeFields(dom, ::Comm::AUX)  // additive ghost->active (Pegasus §3.6)
 *   CommunicateFields(dom, ::Comm::AUX)  // copy active->ghost for the EMF reads
 * and, before each push, fills the bckp (Ec/Bc) ghosts the gather reads:
 *   CommunicateFields(dom, ::Comm::Bckp)
 *
 * TEAM_POLICY: the fused deposit is launched through the generic
 * `kernel::TiledScatter_kernel` harness instead of the flat ScatterView —
 * one team per spatial tile, per-team SLM scratch of (T + 2*HALO)^D x 4,
 * HALO = window + TEAM_POLICY_DRIFT — mirroring the tiled current deposit
 * (engines/srpic/currents.h). Coverage is identical to the tiled currents
 * launcher: flat fallback when the species has no tile layout yet (step 0 /
 * tiny species), flat tail pass over [npart_partitioned, npart) for
 * particles appended since the last sort, per-particle escape valve inside
 * the harness for particles that drifted off their tile.
 *
 * @see kernels/hybrid/pusher.hpp, kernels/tiled_scatter.hpp and
 *      PIC/hybrid/pusher.md.
 */

#ifndef ENGINES_HYBRID_PARTICLE_PUSHER_H
#define ENGINES_HYBRID_PARTICLE_PUSHER_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/log.h"
#include "utils/param_container.h"

#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"
#include "kernels/hybrid/pusher.hpp"
#include "kernels/pushers/context.h" // kernel::sr::PusherBoundaries
#include "kernels/tiled_scatter.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

namespace ntt::hybrid {

  /**
   * @brief Run the fused push+deposit kernel over all ion species into `aux`.
   *        Zeroes aux, deposits N -> aux::3 and V = Σ m v -> aux::0..2.
   *        Caller handles the subsequent ghost sync/comm.
   * @tparam Mode MomentsOnly (no push), Predictor (no store), Corrector (store).
   * @param dt time-step (unused for MomentsOnly).
   * @param team_size_req explicit tiled team size (0 = Kokkos::AUTO); only
   *        read on the TEAM_POLICY path.
   */
  template <kernel::hybrid::PushMode Mode, CartesianMetricClass M>
  void runPusher(Domain<SimEngine::HYBRID, M>& domain,
                 const SimulationParams&       params,
                 real_t                        dt,
                 int                           team_size_req = 0) {
    const auto omegaB0     = params.template get<real_t>("scales.omegaB0");
    const auto inv_n0      = ONE / params.template get<real_t>("scales.n0");
    const auto use_weights = params.template get<bool>("particles.use_weights");

    const auto pusher_boundaries = kernel::sr::PusherBoundaries<M::Dim> {
      domain.mesh.prtl_bc()
    };

    Kokkos::deep_copy(domain.fields.aux, ZERO);

#if !defined(TEAM_POLICY)
    (void)team_size_req;
    auto scatter_aux = Kokkos::Experimental::create_scatter_view(domain.fields.aux);
#endif

    for (auto& species : domain.species) {
      if ((species.npart() == 0) or cmp::AlmostZero_host(species.mass())) {
        continue;
      }
      if constexpr (Mode != kernel::hybrid::PushMode::MomentsOnly) {
        species.set_unsorted();
      }

      const kernel::hybrid::PusherContext ctx {
        species.mass(),
        species.charge(),
        dt,
        omegaB0,
        inv_n0,
        use_weights,
        static_cast<int>(domain.mesh.n_active(in::x1)),
        static_cast<int>(domain.mesh.n_active(in::x2)),
        static_cast<int>(domain.mesh.n_active(in::x3))
      };

#if defined(TEAM_POLICY)
      // Tiled push+deposit. Coverage mirrors the tiled currents launcher
      // (engines/srpic/currents.h): the harness handles a stale partition
      // per-particle (escape valve for drifted particles, dead-tag skip in
      // the kernel, slice clamp to the live npart), the flat tail pass
      // below covers particles appended since the last sort, and species
      // with no tile layout yet (first step, before any SortSpatially)
      // take the flat scatter-view path for that call alone.
      const auto& layout = species.tile_layout();
      if (layout.ntiles_total == 0u or layout.tile_offsets.extent(0) == 0u) {
        auto scatter_aux = Kokkos::Experimental::create_scatter_view(
          domain.fields.aux);
        Kokkos::parallel_for(
          "HybridPushDeposit",
          species.rangeActiveParticles(),
          kernel::hybrid::Pusher_kernel<M, Mode> { ctx,
                                                   pusher_boundaries,
                                                   species,
                                                   domain.fields.bckp,
                                                   scatter_aux,
                                                   domain.mesh.metric });
        Kokkos::Experimental::contribute(domain.fields.aux, scatter_aux);
        continue;
      }

      // Sort-cadence sanity (see the plan's §"sort-cadence gotcha" and the
      // halo derivation in kernels/tiled_scatter.hpp): with an interval
      // K > DRIFT + 1, most particles drift past the scratch halo between
      // sorts and take the global escape valve — correct, but the SLM
      // scratch is silently bypassed and the tiled kernel performs like
      // the flat one with extra overhead. Each push moves ions by <= 1
      // cell (CFL), and the pusher un-sorts them anyway: keep
      // spatial_sorting_interval = 1 for ion species, or build with
      // team_policy_drift = interval.
      {
#if defined(TEAM_POLICY_DRIFT)
        constexpr auto DRIFT = static_cast<timestep_t>(TEAM_POLICY_DRIFT);
#else
        constexpr auto DRIFT = static_cast<timestep_t>(1);
#endif
        static bool warned_cadence = false;
        if ((not warned_cadence) and
            (species.spatial_sorting_interval() > DRIFT + 1u)) {
          warned_cadence = true;
          raise::Warning(
            fmt::format("hybrid tiled deposit: spatial_sorting_interval = %d "
                        "for species %d exceeds team_policy_drift + 1 = %d — "
                        "most particles will bypass the SLM scratch through "
                        "the escape valve; set the interval to 1 or rebuild "
                        "with a larger team_policy_drift",
                        static_cast<int>(species.spatial_sorting_interval()),
                        species.index(),
                        static_cast<int>(DRIFT + 1u)),
            HERE);
        }
      }

      using body_t  = kernel::hybrid::Pusher_kernel<M, Mode>;
      using tiled_t = kernel::TiledScatter_kernel<M::Dim,
                                                  4, // NC: V (0..2) + N (3)
                                                  6, // NG: aux comps
                                                  body_t::window,
                                                  static_cast<unsigned short>(
                                                    TEAM_POLICY_TILE_SIZE),
                                                  body_t>;
      const body_t body { ctx,
                          pusher_boundaries,
                          species,
                          domain.fields.bckp,
                          domain.mesh.metric };
      const tiled_t kern { domain.fields.aux, body, layout, species.npart() };
      const auto    policy = kernel::MakeTiledPolicy(kern,
                                                     layout.ntiles_total,
                                                     team_size_req);
      Kokkos::parallel_for("HybridPushDepositTiled", policy, kern);

      // Particles appended since the last sort (injection / MPI receive on
      // a no-sort step) live past the partition and are not visited by any
      // team above. Push+deposit that tail [npart_partitioned, npart) with
      // the flat scatter-view kernel so every active particle is handled
      // exactly once (the Corrector store-back included).
      if (species.npart() > layout.npart_partitioned) {
        auto scatter_aux = Kokkos::Experimental::create_scatter_view(
          domain.fields.aux);
        Kokkos::parallel_for(
          "HybridPushDepositTail",
          CreateParticleRangePolicy<Dim::_1D>({ layout.npart_partitioned },
                                              { species.npart() }),
          kernel::hybrid::Pusher_kernel<M, Mode> { ctx,
                                                   pusher_boundaries,
                                                   species,
                                                   domain.fields.bckp,
                                                   scatter_aux,
                                                   domain.mesh.metric });
        Kokkos::Experimental::contribute(domain.fields.aux, scatter_aux);
      }
#else
      Kokkos::parallel_for(
        "HybridPushDeposit",
        species.rangeActiveParticles(),
        kernel::hybrid::Pusher_kernel<M, Mode> { ctx,
                                                 pusher_boundaries,
                                                 species,
                                                 domain.fields.bckp,
                                                 scatter_aux,
                                                 domain.mesh.metric });
#endif
    }
#if !defined(TEAM_POLICY)
    Kokkos::Experimental::contribute(domain.fields.aux, scatter_aux);
#endif
  }

  /**
   * @brief Advance all ion species from x^(n),v^(n) to x^(n+1),v^(n+1) and deposit
   *        the corresponding ion moments into `aux` in the same pass.
   * @param corrector false -> transient predictor push (no store, no particle BCs);
   *                  true  -> accepted corrector push (store-back + particle BCs).
   */
  template <CartesianMetricClass M>
  void ParticlePush(Domain<SimEngine::HYBRID, M>& domain,
                    const prm::Parameters&        engine_params,
                    const SimulationParams&       params,
                    bool                          corrector) {
    const auto dt = engine_params.get<real_t>("dt");
    // Optional runtime override for the tiled team (work-group) size;
    // 0 (default) keeps Kokkos::AUTO. Clamped to the backend max in
    // kernel::MakeTiledPolicy. Ignored without TEAM_POLICY.
    const auto team_size_req = static_cast<int>(
      engine_params.get<std::size_t>("team_policy_team_size",
                                     std::optional<std::size_t> { 0u }));
    if (corrector) {
      runPusher<kernel::hybrid::PushMode::Corrector>(domain,
                                                     params,
                                                     dt,
                                                     team_size_req);
    } else {
      runPusher<kernel::hybrid::PushMode::Predictor>(domain,
                                                     params,
                                                     dt,
                                                     team_size_req);
    }
  }

  /**
   * @brief Deposit ion moments N, V from the stored particles into `aux` without
   *        pushing. Used once at step 0 to seed aux with N^(0), V^(0).
   */
  template <CartesianMetricClass M>
  void DepositMoments(Domain<SimEngine::HYBRID, M>& domain,
                      const SimulationParams&       params) {
    runPusher<kernel::hybrid::PushMode::MomentsOnly>(domain, params, ZERO);
  }

} // namespace ntt::hybrid

#endif // ENGINES_HYBRID_PARTICLE_PUSHER_H
