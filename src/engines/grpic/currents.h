/**
 * @file engines/grpic/currents.h
 * @brief Current deposition and filtering routines for the GRPIC engine
 * @implements
 *   - ntt::grpic::CurrentsDeposit<> -> void
 *   - ntt::grpic::CurrentsFilter<> -> void
 * @namespaces:
 *   - ntt::grpic::
 */

#ifndef ENGINES_GRPIC_CURRENTS_H
#define ENGINES_GRPIC_CURRENTS_H

#include "enums.h"

#include "traits/metric.h"
#include "utils/log.h"
#include "utils/param_container.h"

#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "kernels/currents_deposit.hpp"
#include "kernels/digital_filter.hpp"

namespace ntt {
  namespace grpic {

    template <GRMetricClass M>
    void CurrentsDeposit(Domain<SimEngine::GRPIC, M>& domain,
                         const prm::Parameters&       engine_params) {
      auto scatter_cur0 = Kokkos::Experimental::create_scatter_view(
        domain.fields.cur0);
      const auto dt = engine_params.get<real_t>("dt");
      for (auto& species : domain.species) {
        logger::Checkpoint(
          fmt::format("Launching currents deposit kernel for %d [%s] : %lu %f",
                      species.index(),
                      species.label().c_str(),
                      species.npart(),
                      (double)species.charge()),
          HERE);
        if (species.npart() == 0 || cmp::AlmostZero(species.charge())) {
          continue;
        }
        Kokkos::parallel_for("CurrentsDeposit",
                             species.rangeActiveParticles(),
                             kernel::DepositCurrents_kernel<SimEngine::GRPIC, M>(
                               scatter_cur0,
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
                               domain.mesh.metric,
                               (real_t)(species.charge()),
                               dt));
      }
      Kokkos::Experimental::contribute(domain.fields.cur0, scatter_cur0);
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
      tuple_t<std::size_t, M::Dim> size;
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