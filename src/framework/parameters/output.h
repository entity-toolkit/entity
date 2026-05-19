/**
 * @file framework/parameters/output.h
 * @brief Auxiliary functions for reading in output parameters
 * @implements
 *   - ntt::params::Output
 *   - ntt::params::OutputCategory
 * @cpp:
 *   - output.cpp
 * @namespaces:
 *   - ntt::params::
 */
#ifndef FRAMEWORK_PARAMETERS_OUTPUT_H
#define FRAMEWORK_PARAMETERS_OUTPUT_H

#include "global.h"

#include "framework/parameters/parameters.h"

#include <toml11/toml.hpp>

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ntt {
  namespace params {

    struct OutputCategory {
      bool       enable;
      timestep_t interval;
      simtime_t  interval_time;
    };

    struct Output {
      std::optional<std::string> format;
      std::optional<int>         aggregators_per_node;

      std::optional<timestep_t> global_interval;
      std::optional<simtime_t>  global_interval_time;

      std::optional<std::map<std::string, OutputCategory>> categories;

      std::optional<std::vector<std::string>>  fields_quantities;
      std::optional<std::vector<std::string>>  fields_custom_quantities;
      std::optional<unsigned short>            fields_mom_smooth;
      std::optional<std::vector<unsigned int>> fields_downsampling;

      std::optional<std::vector<spidx_t>> particles_species;
      std::optional<npart_t>              particles_stride;

      std::optional<real_t>      spectra_e_min;
      std::optional<real_t>      spectra_e_max;
      std::optional<bool>        spectra_log_bins;
      std::optional<std::size_t> spectra_n_bins;

      std::optional<std::vector<std::string>> stats_quantities;
      std::optional<std::vector<std::string>> stats_custom_quantities;

      std::optional<bool> debug_as_is;
      std::optional<bool> debug_ghosts;

      void read(Dimension, std::size_t, const toml::value&);
      void setParams(SimulationParams*) const;
    };

  } // namespace params
} // namespace ntt

#endif // FRAMEWORK_PARAMETERS_OUTPUT_H
