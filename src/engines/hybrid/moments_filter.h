#ifndef ENGINES_HYBRID_MOMENTS_FILTER_H
#define ENGINES_HYBRID_MOMENTS_FILTER_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/log.h"
#include "utils/param_container.h"

#include "metrics/minkowski.h"

#include "engines/hybrid/fields_bcs.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "kernels/hybrid/moments_filter.hpp"

namespace ntt {
  namespace hybrid {

    /**
     * Binomially smooth the deposited moments (aux::012 = V, aux::3 = N) in
     * place, `algorithms.current_filters` times. Without this, cold/fast beams
     * inject grid-scale shot noise straight into the Ohm's-law E and drive a
     * numerical instability (see kernels/hybrid/moments_filter.hpp).
     *
     * Must be called AFTER the deposit's additive ghost remap
     * (SynchronizeFields(AUX)) and active->ghost copy (CommunicateFields(AUX)),
     * since the stencil reads the i +/- 1 ghosts, which hold valid values to a
     * depth of N_GHOSTS (>= 2) on entry.
     *
     * Passes run in pairs with one halo exchange per pair: the first pass of a
     * pair filters aux -> bckp on the active cells plus one ghost layer on
     * every side refreshed by the halo exchange (periodic / inter-domain), the
     * second filters bckp -> aux on the active cells, reading that layer. A
     * ghost cell filtered locally holds exactly the value the neighbor computes
     * for its own cell (same stencil, ghost copies of the same inputs), i.e.
     * the value a halo exchange would deliver. Wall sides are never extended;
     * their ghosts are mirror-filled after every pass. An odd last pass filters
     * aux in place through a bckp copy.
     *
     * Uses `bckp` as scratch: it is dead at every deposit point in the step
     * (always overwritten by the following EMF before it is read).
     */
    template <Dimension D>
    void MomentsFilter(Metadomain<SimEngine::HYBRID, metric::Minkowski<D>>& metadomain,
                       Domain<SimEngine::HYBRID, metric::Minkowski<D>>&     domain,
                       const SimulationParams&                             params) {
      static_assert(N_GHOSTS >= 2, "paired moments filter needs two ghost layers");
      const auto nfilter = params.template get<unsigned short>(
        "algorithms.current_filters");
      if (nfilter == 0) {
        return;
      }
      logger::Checkpoint("Launching hybrid moments filtering kernels", HERE);

      const auto flds_bc   = domain.mesh.flds_bc();
      const auto comm_side = [](FldsBC b) {
        return (b == FldsBC::PERIODIC) or (b == FldsBC::SYNC);
      };
      bool ext_lo[3] = { false, false, false };
      bool ext_hi[3] = { false, false, false };
      for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
        ext_lo[d] = comm_side(flds_bc[d].first);
        ext_hi[d] = comm_side(flds_bc[d].second);
      }
      // active cells + `m` ghost layers on the comm sides
      const auto range_with_margin = [&](ncells_t m) -> range_t<D> {
        const auto ml = [&](unsigned d) -> ncells_t {
          return ext_lo[d] ? m : 0u;
        };
        const auto mh = [&](unsigned d) -> ncells_t {
          return ext_hi[d] ? m : 0u;
        };
        if constexpr (D == Dim::_1D) {
          return CreateRangePolicy<Dim::_1D>({ domain.mesh.i_min(in::x1) - ml(0) },
                                             { domain.mesh.i_max(in::x1) + mh(0) });
        } else if constexpr (D == Dim::_2D) {
          return CreateRangePolicy<Dim::_2D>(
            { domain.mesh.i_min(in::x1) - ml(0), domain.mesh.i_min(in::x2) - ml(1) },
            { domain.mesh.i_max(in::x1) + mh(0), domain.mesh.i_max(in::x2) + mh(1) });
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

      auto i { 0u };
      for (; i + 1u < nfilter; i += 2u) {
        Kokkos::parallel_for(
          "MomentsFilter",
          range_with_margin(1u),
          kernel::hybrid::MomentsFilter_kernel<D>(domain.fields.bckp,
                                                  domain.fields.aux));
        MomentsWallFill(domain, metadomain.mesh(), domain.fields.bckp);
        Kokkos::parallel_for(
          "MomentsFilter",
          range_with_margin(0u),
          kernel::hybrid::MomentsFilter_kernel<D>(domain.fields.aux,
                                                  domain.fields.bckp));
        metadomain.CommunicateFields(domain, ::Comm::AUX);
        // re-mirror the reflecting-wall ghosts (fill only; the deposit tails
        // were already folded once, before the filter)
        MomentsWallBC(domain, metadomain.mesh(), /* fold */ false);
      }
      if (i < nfilter) {
        Kokkos::deep_copy(domain.fields.bckp, domain.fields.aux);
        Kokkos::parallel_for(
          "MomentsFilter",
          domain.mesh.rangeActiveCells(),
          kernel::hybrid::MomentsFilter_kernel<D>(domain.fields.aux,
                                                  domain.fields.bckp));
        metadomain.CommunicateFields(domain, ::Comm::AUX);
        MomentsWallBC(domain, metadomain.mesh(), /* fold */ false);
      }
    }

  } // namespace hybrid
} // namespace ntt

#endif // ENGINES_HYBRID_MOMENTS_FILTER_H
