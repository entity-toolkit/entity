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
 * @see kernels/hybrid/pusher.hpp and PIC/hybrid/pusher.md.
 */

#ifndef ENGINES_HYBRID_PARTICLE_PUSHER_H
#define ENGINES_HYBRID_PARTICLE_PUSHER_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/comparators.h"
#include "utils/log.h"

#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"
#include "kernels/hybrid/pusher.hpp"
#include "kernels/pushers/context.h" // kernel::sr::PusherBoundaries

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

namespace ntt::hybrid {

  /**
   * @brief Run the fused push+deposit kernel over all ion species into `aux`.
   *        Zeroes aux, scatter-deposits N -> aux::3 and V = Σ m v -> aux::0..2,
   *        and contributes. Caller handles the subsequent ghost sync/comm.
   * @tparam Mode MomentsOnly (no push), Predictor (no store), Corrector (store).
   * @param dt time-step (unused for MomentsOnly).
   */
  template <kernel::hybrid::PushMode Mode, CartesianMetricClass M>
  void runPusher(Domain<SimEngine::HYBRID, M>& domain,
                 const SimulationParams&       params,
                 real_t                        dt) {
    const auto omegaB0     = params.template get<real_t>("scales.omegaB0");
    const auto inv_n0      = ONE / params.template get<real_t>("scales.n0");
    const auto use_weights = params.template get<bool>("particles.use_weights");

    const auto pusher_boundaries = kernel::sr::PusherBoundaries<M::Dim> {
      domain.mesh.prtl_bc()
    };

    Kokkos::deep_copy(domain.fields.aux, ZERO);
    auto scatter_aux = Kokkos::Experimental::create_scatter_view(domain.fields.aux);

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

      Kokkos::parallel_for(
        "HybridPushDeposit",
        species.rangeActiveParticles(),
        kernel::hybrid::Pusher_kernel<M, Mode> { ctx,
                                                 pusher_boundaries,
                                                 species,
                                                 domain.fields.bckp,
                                                 scatter_aux,
                                                 domain.mesh.metric });
    }
    Kokkos::Experimental::contribute(domain.fields.aux, scatter_aux);
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
    if (corrector) {
      runPusher<kernel::hybrid::PushMode::Corrector>(domain, params, dt);
    } else {
      runPusher<kernel::hybrid::PushMode::Predictor>(domain, params, dt);
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
