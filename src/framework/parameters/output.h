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

#include <map>
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
      std::string format;

      timestep_t global_interval;
      simtime_t  global_interval_time;

      std::map<std::string, OutputCategory> categories;

      std::vector<std::string>  fields_quantities;
      std::vector<std::string>  fields_custom_quantities;
      unsigned short            fields_mom_smooth;
      std::vector<unsigned int> fields_downsampling;

      std::vector<spidx_t> particles_species;
      npart_t              particles_stride;

      real_t      spectra_e_min;
      real_t      spectra_e_max;
      bool        spectra_log_bins;
      std::size_t spectra_n_bins;

      real_t spectra3D_e_min;
      real_t spectra3D_e_max;
      bool spectra3D_log_bins;
      std::size_t spectra3D_n_bins;
      std::size_t spectra3D_nx1;
      std::size_t spectra3D_nx2;
      std::size_t spectra3D_nx3;

      std::vector<std::string> stats_quantities;
      std::vector<std::string> stats_custom_quantities;

      bool debug_as_is;
      bool debug_ghosts;

      void read(Dimension, std::size_t, const toml::value&);
      void setParams(SimulationParams*) const;
    };

  } // namespace params
} // namespace ntt

#endif // FRAMEWORK_PARAMETERS_OUTPUT_H
