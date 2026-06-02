#ifndef ENGINES_HYBRID_FIELDSOLVERS_H
#define ENGINES_HYBRID_FIELDSOLVERS_H

#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/param_container.h"

#include "metrics/minkowski.h"

#include "framework/domain/domain.h"
#include "kernels/hybrid/faraday.hpp"

namespace ntt {
  namespace hybrid {

    enum class faraday : uint8_t {
      push1,
      push2,
      push3,
    };

    template <Dimension D>
    void Faraday(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                 const prm::Parameters&                           engine_params,
                 const faraday&                                   flag) {
      const auto dt = engine_params.get<real_t>("dt");
      if (flag == faraday::push1) {
        Kokkos::parallel_for(
          "FaradayPush1",
          domain.mesh.rangeActiveCells(),
          kernel::hybrid::Faraday_kernel<D, 3>(domain.fields.em,
                                               domain.fields.em,
                                               domain.fields.cur,
                                               0,
                                               3,
                                               0,
                                               dt));
      } else if (flag == faraday::push2) {
        Kokkos::parallel_for(
          "FaradayPush2",
          domain.mesh.rangeActiveCells(),
          kernel::hybrid::Faraday_kernel<D, 3>(domain.fields.em0,
                                               domain.fields.em,
                                               domain.fields.cur,
                                               3,
                                               3,
                                               0,
                                               dt));
      } else if (flag == faraday::push3) {
        Kokkos::parallel_for(
          "FaradayPush3",
          domain.mesh.rangeActiveCells(),
          kernel::hybrid::Faraday_kernel<D, 6>(domain.fields.em0,
                                               domain.fields.em,
                                               domain.fields.em,
                                               3,
                                               3,
                                               3,
                                               dt));
      } else {
        raise::Error("Wrong option for `flag`", HERE);
      }
    }

  } // namespace hybrid
} // namespace ntt

#endif
