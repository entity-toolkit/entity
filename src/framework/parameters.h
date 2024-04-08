/**
 * @file framework/parameters.h
 * @brief Structure for defining and holding initial simulation parameters
 * @implements
 *   - ntt::SimulationParams : ntt::Parameters
 * @depends:
 *   - utils/param_container.h
 *   - defaults.h
 *   - enums.h
 *   - arch/kokkos_aliases.h
 *   - framework/species.h
 *   - utils/error.h
 *   - utils/formatting.h
 *   - utils/log.h
 *   - utils/numeric.h
 *   - metrics/kerr_schild.h
 *   - metrics/minkowski.h
 *   - metrics/qkerr_schild.h
 *   - metrics/qspherical.h
 *   - metrics/spherical.h
 *   - metrics/kerr_schild_0.h
 * @cpp:
 *   - parameters.cpp
 * @namespaces:
 *   - ntt::
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
    SimulationParams(const toml::value&);
    ~SimulationParams() = default;

  private:
    const toml::value raw_data;
  };

} // namespace ntt

#endif // FRAMEWORK_PARAMETERS_H
