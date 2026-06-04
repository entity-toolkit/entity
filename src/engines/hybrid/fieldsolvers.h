#ifndef ENGINES_HYBRID_FIELDSOLVERS_H
#define ENGINES_HYBRID_FIELDSOLVERS_H

#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/param_container.h"

#include "metrics/minkowski.h"

#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"
#include "kernels/hybrid/EMF.hpp"
#include "kernels/hybrid/faraday.hpp"

namespace ntt {
  namespace hybrid {

    enum class faraday : uint8_t {
      push1,
      push2,
      push3,
    };

    enum class emf : uint8_t {
      push0,
      push12,
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
             const prm::Parameters&                           engine_params,
             const SimulationParams&                          params,
             const emf&                                       flag) {
      const auto dt       = engine_params.get<real_t>("dt");
      const auto gamma_ad = params.get<real_t>("scales.gamma_ad");
      const auto theta    = params.get<real_t>("scales.theta0");
      const auto d0       = params.get<real_t>("scales.skindepth0");
      const auto rho0     = params.get<real_t>("scales.larmor0");
      if (flag == emf::push0) {
        // clang-format off
        Kokkos::parallel_for(
          "EMFPush",
          domain.mesh.rangeActiveCells(),
          kernel::hybrid::EMF_kernel<D, true>(domain.fields.aux,  // P
                                              domain.fields.aux,  // N
                                              domain.fields.em0,  // Ee_in
                                              domain.fields.cur,  // Bf
                                              domain.fields.bckp, // Ec
                                              domain.fields.em,   // Bfs
                                              domain.fields.em,   // Ee_out
                                              domain.fields.em0,  // Ec_out
                                              domain.fields.bckp, // Bc_out
                                              0, 3, 3,            // P, N, Ee_in
                                              0, 0, 3,            // Bf, Ec, Bfs
                                              0, 0, 3,            // Ee_out, Ec_out, Bc_out
                                              dt, gamma_ad, theta, d0, rho0));
        // clang-format on
      } else {
        // clang-format off
        Kokkos::parallel_for(
          "EMFPush",
          domain.mesh.rangeActiveCells(),
          kernel::hybrid::EMF_kernel<D, false>(domain.fields.aux,  // P
                                               domain.fields.aux,  // N
                                               domain.fields.em,   // Ee_in
                                               domain.fields.em,   // Bf
                                               domain.fields.em0,  // Ec
                                               domain.fields.cur,  // Bfs
                                               domain.fields.em0,  // Ee_out
                                               domain.fields.bckp, // Ec_out
                                               domain.fields.bckp, // Bc_out
                                               0, 3, 0,            // P, N, Ee_in
                                               3, 0, 0,            // Bf, Ec, Bfs
                                               3, 0, 3,            // Ee_out, Ec_out, Bc_out
                                               dt, gamma_ad, theta, d0, rho0));
        // clang-format on
      }
    }

  } // namespace hybrid
} // namespace ntt

#endif
