#ifndef ENGINES_HYBRID_FIELDSOLVERS_H
#define ENGINES_HYBRID_FIELDSOLVERS_H

#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/param_container.h"

#include "metrics/minkowski.h"

#include "framework/domain/domain.h"
#include "kernels/hybrid/EMF.hpp"
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

    template <Dimension D>
    void EMF(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
             const prm::Parameters&                           engine_params) {
      const auto dt       = engine_params.get<real_t>("dt");
      const auto gamma_ad = engine_params.get<real_t>("gamma_ad");
      const auto theta    = engine_params.get<real_t>("theta");
      const auto d0       = engine_params.get<real_t>("skindepth0");
      const auto rho0     = engine_params.get<real_t>("larmor0");
      Kokkos::parallel_for(
        "EMFPush1",
        domain.mesh.rangeActiveCells(),
        kernel::hybrid::Faraday_kernel<D, 3>(domain.fields.aux,  // P
                                             domain.fields.aux,  // N
                                             domain.fields.em,   // Ee_in
                                             domain.fields.em,   // Bf
                                             domain.fields.em0,  // Ec
                                             domain.fields.cur,  // Bfs
                                             domain.fields.em0,  // Ee_out
                                             domain.fields.bckp, // Ec_out
                                             domain.fields.bckp, // Bc_out
                                             0,                  // P
                                             3,                  // N
                                             0,                  // Ee_in
                                             3,                  // Bf
                                             0,                  // Ec
                                             0,                  // Bfs
                                             3,                  // Ee_out
                                             0,                  // Ec_out
                                             3,                  // Bc_out
                                             dt,
                                             gamma_ad,
                                             theta,
                                             d0,
                                             rho0));
    }

  } // namespace hybrid
} // namespace ntt

#endif
