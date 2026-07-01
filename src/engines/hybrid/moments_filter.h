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
     * since the stencil reads the i +/- 1 ghosts; each pass re-fills them.
     *
     * Uses `bckp` as the read buffer: it is dead scratch at every deposit point
     * in the step (always overwritten by the following EMF before it is read).
     */
    template <Dimension D>
    void MomentsFilter(Metadomain<SimEngine::HYBRID, metric::Minkowski<D>>& metadomain,
                       Domain<SimEngine::HYBRID, metric::Minkowski<D>>&     domain,
                       const SimulationParams&                             params) {
      const auto nfilter = params.template get<unsigned short>(
        "algorithms.current_filters");
      if (nfilter == 0) {
        return;
      }
      logger::Checkpoint("Launching hybrid moments filtering kernels", HERE);
      for (auto i { 0u }; i < nfilter; ++i) {
        Kokkos::deep_copy(domain.fields.bckp, domain.fields.aux);
        Kokkos::parallel_for(
          "MomentsFilter",
          domain.mesh.rangeActiveCells(),
          kernel::hybrid::MomentsFilter_kernel<D>(domain.fields.aux,
                                                  domain.fields.bckp));
        metadomain.CommunicateFields(domain, ::Comm::AUX);
        // re-mirror the reflecting-wall ghosts (fill only; the deposit tails
        // were already folded once, before the filter)
        MomentsWallBC(domain, metadomain.mesh(), /* fold */ false);
      }
    }

  } // namespace hybrid
} // namespace ntt

#endif // ENGINES_HYBRID_MOMENTS_FILTER_H
