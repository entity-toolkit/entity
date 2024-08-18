/**
 * @file framework/parameters.h
 * @brief Structure for defining and holding initial simulation parameters
 * @implements
 *   - ntt::SimulationParams : ntt::Parameters
 * @cpp:
 *   - parameters.cpp
 * @namespaces:
 *   - ntt::
 * @macros:
 *   - MPI_ENABLED
 * @note The parameters are read from the toml file and stored in a container
 * @note Some of the parameters are inferred from the context (see input.example.toml)
 * @note A proper metric is used to infer the minimum cell size/volume etc.
 */

#ifndef FRAMEWORK_PARAMETERS_H
#define FRAMEWORK_PARAMETERS_H

#include "utils/param_container.h"

#include <toml.hpp>

namespace ntt {

  struct SimulationParams : public prm::Parameters {
    SimulationParams() = default;
    SimulationParams(const toml::value&);

    SimulationParams& operator=(const SimulationParams& other) {
      vars     = std::move(other.vars);
      promises = std::move(other.promises);
      return *this;
    }

    ~SimulationParams() = default;
  };

} // namespace ntt

#endif // FRAMEWORK_PARAMETERS_H
