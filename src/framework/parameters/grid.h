/**
 * @file framework/parameters/grid.h
 * @brief Auxiliary functions for reading in grid/box parameters
 * @implements
 *   - ntt::params::Boundaries
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

#include "framework/parameters/parameters.h"

#include <toml11/toml.hpp>

#include <map>
#include <optional>
#include <tuple>
#include <vector>

namespace ntt {
  namespace params {

    struct Boundaries {
      const bool                          needs_match_boundaries;
      std::optional<boundaries_t<real_t>> match_ds_array;

      const bool            needs_absorb_boundaries;
      std::optional<real_t> absorb_ds;

      const bool                                 needs_atmosphere_boundaries;
      std::optional<real_t>                      atmosphere_temperature;
      std::optional<real_t>                      atmosphere_height;
      std::optional<real_t>                      atmosphere_density;
      std::optional<real_t>                      atmosphere_g;
      std::optional<real_t>                      atmosphere_ds;
      std::optional<std::pair<spidx_t, spidx_t>> atmosphere_species;

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

    struct Grid {
      std::optional<unsigned int>     number_of_domains;
      std::optional<std::vector<int>> domain_decomposition;

      std::optional<std::vector<ncells_t>> resolution;
      std::optional<Dimension>             dim;

      std::optional<std::vector<std::vector<real_t>>> extent;
      std::optional<boundaries_t<real_t>>             extent_pairwise_;

      Metric metric_enum = Metric::INVALID;
      Coord  coord_enum  = Coord::INVALID;
      std::optional<std::map<std::string, real_t>> metric_params;
      std::optional<std::map<std::string, real_t>> metric_params_short_;

      std::optional<real_t> scale_dx0;
      std::optional<real_t> scale_V0;

      void read(const SimEngine&, const toml::value&);
      void setParams(SimulationParams*) const;
    };

    auto GetBoundaryConditions(SimulationParams* params,
                               const SimEngine&,
                               Dimension,
                               const Coord&,
                               const toml::value&)
      -> std::tuple<boundaries_t<FldsBC>, boundaries_t<PrtlBC>>;

  } // namespace params
} // namespace ntt

#endif // FRAMEWORK_PARAMETERS_GRID_H
