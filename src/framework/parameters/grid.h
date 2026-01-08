#ifndef FRAMEWORK_PARAMETERS_GRID_H
#define FRAMEWORK_PARAMETERS_GRID_H

#include "enums.h"
#include "global.h"

#include "utils/toml.h"

#include "framework/parameters/parameters.h"

#include <map>
#include <tuple>
#include <vector>

namespace ntt {
  namespace params {

    auto GetGridParams(
      const toml::value&) -> std::tuple<std::vector<ncells_t>, Dimension>;

    auto GetMetricParams(const SimEngine&, Dimension, const toml::value&)
      -> std::tuple<Metric, Coord, std::map<std::string, real_t>>;

    auto GetBoundaryConditions(
      SimulationParams* params,
      const SimEngine&,
      Dimension,
      const Coord&,
      const toml::value&) -> std::tuple<boundaries_t<FldsBC>, boundaries_t<PrtlBC>>;

  } // namespace params
} // namespace ntt

#endif // FRAMEWORK_PARAMETERS_GRID_H
