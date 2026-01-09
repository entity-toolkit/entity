/**
 * @file framework/parameters/grid.h
 * @brief Auxiliary functions for reading in grid/box parameters
 * @implements
 *   - ntt::params::Boundaries
 *   - ntt::params::GetGridParams -> (std::vector<ncells_t>, Dimension)
 *   - ntt::params::GetMetricParams -> (Metric, Coord, std::map<std::string, real_t>)
 *   - ntt::params::GetBoundaryConditions -> (boundaries_t<FldsBC>, boundaries_t<PrtlBC>)
 * @cpp:
 *   - grid.cpp
 * @namespaces:
 *   - ntt::params::
 */
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

    struct Boundaries {
      const bool           needs_match_boundaries;
      boundaries_t<real_t> match_ds_array;

      const bool needs_absorb_boundaries;
      real_t     absorb_ds;

      const bool                  needs_atmosphere_boundaries;
      real_t                      atmosphere_temperature;
      real_t                      atmosphere_height;
      real_t                      atmosphere_density;
      real_t                      atmosphere_g;
      real_t                      atmosphere_ds;
      std::pair<spidx_t, spidx_t> atmosphere_species;

      Boundaries(bool needs_match, bool needs_absorb, bool needs_atmosphere)
        : needs_match_boundaries { needs_match }
        , needs_absorb_boundaries { needs_absorb }
        , needs_atmosphere_boundaries { needs_atmosphere } {}

      void read(Dimension,
                const Coord&,
                const boundaries_t<real_t>&,
                const toml::value&);
      void setParams(SimulationParams*) const;
    };

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
