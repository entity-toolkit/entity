/**
 * @file framework/parameters/algorithms.h
 * @brief Auxiliary functions for reading in algorithms parameters
 * @implements
 *   - ntt::params::Algorithms
 * @cpp:
 *   - algorithms.cpp
 * @namespaces:
 *   - ntt::params::
 */

#ifndef FRAMEWORK_PARAMETERS_ALGORITHMS_H
#define FRAMEWORK_PARAMETERS_ALGORITHMS_H

#include "global.h"

#include "framework/parameters/parameters.h"

#include <toml11/toml.hpp>

#include <map>
#include <optional>
#include <string>

namespace ntt {
  namespace params {

    struct Algorithms {
      std::optional<real_t> CFL;
      std::optional<real_t> dt;
      std::optional<real_t> dt_correction_factor;

      std::optional<unsigned short> number_of_current_filters;

      std::optional<bool>           deposit_enable;
      std::optional<unsigned short> deposit_order;

      std::optional<bool>                          fieldsolver_enable;
      std::optional<std::map<std::string, real_t>> fieldsolver_stencil_coeffs;

      std::optional<real_t>         gr_pusher_eps;
      std::optional<unsigned short> gr_pusher_niter;

      std::optional<real_t> gca_e_ovr_b_max;
      std::optional<real_t> gca_larmor_max;

      std::optional<real_t> synchrotron_gamma_rad;
      std::optional<real_t> compton_gamma_rad;

      void read(real_t, const std::map<std::string, bool>&, const toml::value&);
      void setParams(const std::map<std::string, bool>&, SimulationParams*) const;
    };

  } // namespace params
} // namespace ntt

#endif // FRAMEWORK_PARAMETERS_ALGORITHMS_H
