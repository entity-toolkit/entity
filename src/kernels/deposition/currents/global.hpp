/**
 * @file kernels/deposition/currents/global.hpp
 * @brief Global current deposition kernel without tiling.
 *
 * @implements
 *   - kernel::DepositCurrents_kernel<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_DEPOSITION_CURRENTS_GLOBAL_HPP
#define KERNELS_DEPOSITION_CURRENTS_GLOBAL_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "kernels/deposition/currents/single-particle.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

namespace kernel {
  using namespace ntt;

  /**
   * @brief Flat current-deposition kernel.
   *
   * One thread per particle (RangePolicy). Writes are coalesced through a
   * `Kokkos::Experimental::ScatterView` to avoid per-thread atomics on
   * global J. Constructor signature is unchanged from prior versions —
   * `engines/srpic/currents.h` continues to call it identically.
   */
  template <SimEngine::type S, MetricClass M, unsigned short O = 1u>
  class DepositCurrents_kernel {
    static_assert(O <= 11u, "Shape function order O must be <= 11");
    static constexpr auto D = M::Dim;

    scatter_ndfield_t<D, 3> J;
    const ParticleArrays    prtls;
    const M                 metric;
    const real_t            charge, inv_dt;

  public:
    DepositCurrents_kernel(const scatter_ndfield_t<D, 3>& scatter_cur,
                           const ParticleArrays&          prtls,
                           const M&                       metric,
                           real_t                         charge,
                           const real_t                   dt)
      : J { scatter_cur }
      , prtls { prtls }
      , metric { metric }
      , charge { charge }
      , inv_dt { ONE / dt } {
      raise::ErrorIf(
        (O == 2u and N_GHOSTS < 2),
        "Order of interpolation is 2, but number of ghost cells is < 2",
        HERE);
    }

    Inline auto operator()(prtlidx_t p) const -> void {
      auto J_acc = J.access();
      if constexpr (D == Dim::_1D) {
        DepositOneParticle<S, M, O>(p,
                                    prtls,
                                    metric,
                                    charge,
                                    inv_dt,
                                    [&](int g_i1, int comp, real_t v) {
                                      J_acc(g_i1, comp) += v;
                                    });
      } else if constexpr (D == Dim::_2D) {
        DepositOneParticle<S, M, O>(p,
                                    prtls,
                                    metric,
                                    charge,
                                    inv_dt,
                                    [&](int g_i1, int g_i2, int comp, real_t v) {
                                      J_acc(g_i1, g_i2, comp) += v;
                                    });
      } else if constexpr (D == Dim::_3D) {
        DepositOneParticle<S, M, O>(
          p,
          prtls,
          metric,
          charge,
          inv_dt,
          [&](int g_i1, int g_i2, int g_i3, int comp, real_t v) {
            J_acc(g_i1, g_i2, g_i3, comp) += v;
          });
      }
    }
  };

} // namespace kernel

#endif // KERNELS_DEPOSITION_CURRENTS_GLOBAL_HPP
