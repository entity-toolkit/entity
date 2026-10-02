/**
 * @file framework/parameters/extra.h
 * @brief Auxiliary functions for reading in extra physics parameters
 * @implements
 *   - ntt::params::Extra
 * @cpp:
 *   - extra.cpp
 * @namespaces:
 *   - ntt::params::
 */

#ifndef FRAMEWORK_PARAMETERS_EXTRA_H
#define FRAMEWORK_PARAMETERS_EXTRA_H

#include "enums.h"
#include "global.h"

#include "framework/parameters/parameters.h"

#include <toml11/toml.hpp>

#include <map>
#include <optional>
#include <string>

namespace ntt {
  namespace params {

    struct TwoBodyInteractionParams {
      TwoBodyInteractionFlag type;
      std::vector<spidx_t>   group1;
      std::vector<spidx_t>   group2;
      timestep_t             interval;
      ncells_t               tile_size;
      bool                   recoil1;
      bool                   recoil2;
    };

    struct Extra {
      // radiative drag parameters
      std::optional<real_t> synchrotron_gamma_rad;
      std::optional<real_t> compton_gamma_rad;

      // emission parameters
      std::optional<real_t>  synchrotron_energy_min;
      std::optional<real_t>  synchrotron_gamma_qed;
      std::optional<real_t>  synchrotron_photon_weight;
      std::optional<spidx_t> synchrotron_photon_species;
      std::optional<real_t>  synchrotron_nominal_probability;
      std::optional<real_t>  synchrotron_nominal_photon_energy;

      std::optional<real_t>  compton_energy_min;
      std::optional<real_t>  compton_gamma_qed;
      std::optional<real_t>  compton_photon_weight;
      std::optional<spidx_t> compton_photon_species;
      std::optional<real_t>  compton_nominal_probability;
      std::optional<real_t>  compton_nominal_photon_energy;

      // two-body interaction parameters
      std::optional<real_t>                 twobody_thomson_optical_depth;
      std::vector<TwoBodyInteractionParams> twobody_interactions;

      void read(const std::map<std::string, bool>&,
                const toml::value&,
                const SimulationParams* const);
      void setParams(const std::map<std::string, bool>&, SimulationParams*) const;
    };

  } // namespace params
} // namespace ntt

#endif // FRAMEWORK_PARAMETERS_EXTRA_H
