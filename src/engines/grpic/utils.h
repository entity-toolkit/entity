/**
 * @file engines/grpic/utils.h
 * @brief Utility functions used by the GRPIC engine
 * @implements
 *   - ntt::grpic::SwapFields<> -> void
 *   - ntt::grpic::CopyFields<> -> void
 *   - ntt::grpic::TimeAverageDB<> -> void
 *   - ntt::grpic::TimeAverageJ<> -> void
 * @namespaces:
 *   - ntt::grpic::
 */

#ifndef ENGINES_GRPIC_UTILS_H
#define ENGINES_GRPIC_UTILS_H

#include "enums.h"

#include "traits/metric.h"

#include "framework/domain/domain.h"
#include "kernels/aux_fields_gr.hpp"

#include <utility>

namespace ntt {
  namespace grpic {

    /**
     * @brief Swaps em and em0 fields, cur and cur0 currents.
     */
    template <GRMetricClass M>
    void SwapFields(Domain<SimEngine::GRPIC, M>& domain) {
      std::swap(domain.fields.em, domain.fields.em0);
      std::swap(domain.fields.cur, domain.fields.cur0);
    }

    /**
     * @brief Copies em fields into em0
     */
    template <GRMetricClass M>
    void CopyFields(Domain<SimEngine::GRPIC, M>& domain) {
      Kokkos::deep_copy(domain.fields.em0, domain.fields.em);
    }

    template <GRMetricClass M>
    void TimeAverageDB(Domain<SimEngine::GRPIC, M>& domain) {
      Kokkos::parallel_for(
        "TimeAverageDB",
        domain.mesh.rangeActiveCells(),
        kernel::gr::TimeAverageDB_kernel<M::Dim>(domain.fields.em,
                                                 domain.fields.em0));
    }

    template <GRMetricClass M>
    void TimeAverageJ(Domain<SimEngine::GRPIC, M>& domain) {
      Kokkos::parallel_for(
        "TimeAverageJ",
        domain.mesh.rangeActiveCells(),
        kernel::gr::TimeAverageJ_kernel<M::Dim>(domain.fields.cur,
                                                domain.fields.cur0));
    }

  } // namespace grpic
} // namespace ntt

#endif // ENGINES_GRPIC_UTILS_H