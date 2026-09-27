/**
 * @file framework/parameters/render.h
 * @brief Auxiliary functions for reading in on-the-fly render parameters
 * @implements
 *   - ntt::params::Render
 * @cpp:
 *   - render.cpp
 * @namespaces:
 *   - ntt::params::
 */
#ifndef FRAMEWORK_PARAMETERS_RENDER_H
#define FRAMEWORK_PARAMETERS_RENDER_H

#include "global.h"

#include "framework/parameters/parameters.h"

#include <toml11/toml.hpp>

#include <map>
#include <optional>

namespace ntt {
  namespace params {

    struct Render {

      void read(const std::map<std::string, bool>&,
                const toml::value&,
                const SimulationParams* const);
      void setParams(const std::map<std::string, bool>&, SimulationParams*) const;
    };

  } // namespace params
} // namespace ntt

#endif
