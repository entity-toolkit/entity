#include "framework/parameters/grid.h"

#include "defaults.h"
#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/numeric.h"
#include "utils/toml.h"

#include "framework/parameters/parameters.h"

#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace ntt {
  namespace params {

    auto GetGridParams(
      const toml::value& toml_data) -> std::tuple<std::vector<ncells_t>, Dimension> {
      const auto res = toml::find<std::vector<ncells_t>>(toml_data,
                                                         "grid",
                                                         "resolution");
      raise::ErrorIf(res.size() < 1 || res.size() > 3,
                     "invalid `grid.resolution`",
                     HERE);
      const auto dim = static_cast<Dimension>(res.size());
      return { res, dim };
    }

    auto GetMetricParams(const SimEngine&   engine_enum,
                         Dimension          dim,
                         const toml::value& toml_data)
      -> std::tuple<Metric, Coord, std::map<std::string, real_t>> {
      const auto metric_enum = Metric::pick(
        fmt::toLower(toml::find<std::string>(toml_data, "grid", "metric", "metric"))
          .c_str());
      std::map<std::string, real_t> additional_params;
      std::string                   coord;
      if (metric_enum == Metric::Minkowski) {
        raise::ErrorIf(engine_enum != SimEngine::SRPIC,
                       "minkowski metric is only supported for SRPIC",
                       HERE);
        coord = "cart";
      } else if (metric_enum == Metric::QKerr_Schild or
                 metric_enum == Metric::QSpherical) {
        // quasi-spherical geometry
        raise::ErrorIf(dim == Dim::_1D,
                       "not enough dimensions for qspherical geometry",
                       HERE);
        raise::ErrorIf(dim == Dim::_3D,
                       "3D not implemented for qspherical geometry",
                       HERE);
        coord                        = "qsph";
        additional_params["qsph_r0"] = toml::find_or(toml_data,
                                                     "grid",
                                                     "metric",
                                                     "qsph_r0",
                                                     defaults::qsph::r0);
        additional_params["qsph_h"]  = toml::find_or(toml_data,
                                                    "grid",
                                                    "metric",
                                                    "qsph_h",
                                                    defaults::qsph::h);
      } else {
        // spherical geometry
        raise::ErrorIf(dim == Dim::_1D,
                       "not enough dimensions for spherical geometry",
                       HERE);
        raise::ErrorIf(dim == Dim::_3D,
                       "3D not implemented for spherical geometry",
                       HERE);
        coord = "sph";
      }
      if ((engine_enum == SimEngine::GRPIC) &&
          (metric_enum != Metric::Kerr_Schild_0)) {
        const auto ks_a            = toml::find_or(toml_data,
                                        "grid",
                                        "metric",
                                        "ks_a",
                                        defaults::ks::a);
        additional_params["ks_a"]  = ks_a;
        additional_params["ks_rh"] = ONE + math::sqrt(ONE - SQR(ks_a));
      }
      const auto coord_enum = Coord::pick(coord.c_str());
      return { metric_enum, coord_enum, additional_params };
    }

    auto GetBoundaryConditions(SimulationParams*  params,
                               const SimEngine&   engine_enum,
                               Dimension          dim,
                               const Coord&       coord_enum,
                               const toml::value& toml_data)
      -> std::tuple<boundaries_t<FldsBC>, boundaries_t<PrtlBC>> {
      auto flds_bc = toml::find<std::vector<std::vector<std::string>>>(
        toml_data,
        "grid",
        "boundaries",
        "fields");
      {
        raise::ErrorIf(flds_bc.size() < 1 || flds_bc.size() > 3,
                       "invalid `grid.boundaries.fields`",
                       HERE);
        params->promiseToDefine("grid.boundaries.fields");
        auto atm_defined = false;
        for (const auto& bcs : flds_bc) {
          for (const auto& bc : bcs) {
            if (fmt::toLower(bc) == "match") {
              params->promiseToDefine("grid.boundaries.match.ds");
            }
            if (fmt::toLower(bc) == "atmosphere") {
              raise::ErrorIf(atm_defined,
                             "ATMOSPHERE is only allowed in one direction",
                             HERE);
              atm_defined = true;
              params->promiseToDefine("grid.boundaries.atmosphere.temperature");
              params->promiseToDefine("grid.boundaries.atmosphere.density");
              params->promiseToDefine("grid.boundaries.atmosphere.height");
              params->promiseToDefine("grid.boundaries.atmosphere.ds");
              params->promiseToDefine("grid.boundaries.atmosphere.species");
              params->promiseToDefine("grid.boundaries.atmosphere.g");
            }
          }
        }
      }

      auto prtl_bc = toml::find<std::vector<std::vector<std::string>>>(
        toml_data,
        "grid",
        "boundaries",
        "particles");
      {
        raise::ErrorIf(prtl_bc.size() < 1 || prtl_bc.size() > 3,
                       "invalid `grid.boundaries.particles`",
                       HERE);
        params->promiseToDefine("grid.boundaries.particles");
        auto atm_defined = false;
        for (const auto& bcs : prtl_bc) {
          for (const auto& bc : bcs) {
            if (fmt::toLower(bc) == "absorb") {
              params->promiseToDefine("grid.boundaries.absorb.ds");
            }
            if (fmt::toLower(bc) == "atmosphere") {
              raise::ErrorIf(atm_defined,
                             "ATMOSPHERE is only allowed in one direction",
                             HERE);
              atm_defined = true;
              params->promiseToDefine("grid.boundaries.atmosphere.temperature");
              params->promiseToDefine("grid.boundaries.atmosphere.density");
              params->promiseToDefine("grid.boundaries.atmosphere.height");
              params->promiseToDefine("grid.boundaries.atmosphere.ds");
              params->promiseToDefine("grid.boundaries.atmosphere.species");
              params->promiseToDefine("grid.boundaries.atmosphere.g");
            }
          }
        }
      }
      std::vector<std::vector<FldsBC>> flds_bc_enum;
      std::vector<std::vector<PrtlBC>> prtl_bc_enum;
      if (coord_enum == Coord::Cart) {
        raise::ErrorIf(flds_bc.size() != (std::size_t)dim,
                       "invalid `grid.boundaries.fields`",
                       HERE);
        raise::ErrorIf(prtl_bc.size() != (std::size_t)dim,
                       "invalid `grid.boundaries.particles`",
                       HERE);
        for (auto d { 0u }; d < (dim_t)dim; ++d) {
          flds_bc_enum.push_back({});
          prtl_bc_enum.push_back({});
          const auto fbc = flds_bc[d];
          const auto pbc = prtl_bc[d];
          raise::ErrorIf(fbc.size() < 1 || fbc.size() > 2,
                         "invalid `grid.boundaries.fields`",
                         HERE);
          raise::ErrorIf(pbc.size() < 1 || pbc.size() > 2,
                         "invalid `grid.boundaries.particles`",
                         HERE);
          auto fbc_enum = FldsBC::pick(fmt::toLower(fbc[0]).c_str());
          auto pbc_enum = PrtlBC::pick(fmt::toLower(pbc[0]).c_str());
          if (fbc.size() == 1) {
            raise::ErrorIf(fbc_enum != FldsBC::PERIODIC,
                           "invalid `grid.boundaries.fields`",
                           HERE);
            flds_bc_enum.back().push_back(FldsBC(FldsBC::PERIODIC));
            flds_bc_enum.back().push_back(FldsBC(FldsBC::PERIODIC));
          } else {
            raise::ErrorIf(fbc_enum == FldsBC::PERIODIC,
                           "invalid `grid.boundaries.fields`",
                           HERE);
            flds_bc_enum.back().push_back(fbc_enum);
            auto fbc_enum = FldsBC::pick(fmt::toLower(fbc[1]).c_str());
            raise::ErrorIf(fbc_enum == FldsBC::PERIODIC,
                           "invalid `grid.boundaries.fields`",
                           HERE);
            flds_bc_enum.back().push_back(fbc_enum);
          }
          if (pbc.size() == 1) {
            raise::ErrorIf(pbc_enum != PrtlBC::PERIODIC,
                           "invalid `grid.boundaries.particles`",
                           HERE);
            prtl_bc_enum.back().push_back(PrtlBC(PrtlBC::PERIODIC));
            prtl_bc_enum.back().push_back(PrtlBC(PrtlBC::PERIODIC));
          } else {
            raise::ErrorIf(pbc_enum == PrtlBC::PERIODIC,
                           "invalid `grid.boundaries.particles`",
                           HERE);
            prtl_bc_enum.back().push_back(pbc_enum);
            auto pbc_enum = PrtlBC::pick(fmt::toLower(pbc[1]).c_str());
            raise::ErrorIf(pbc_enum == PrtlBC::PERIODIC,
                           "invalid `grid.boundaries.particles`",
                           HERE);
            prtl_bc_enum.back().push_back(pbc_enum);
          }
        }
      } else {
        raise::ErrorIf(flds_bc.size() > 1, "invalid `grid.boundaries.fields`", HERE);
        raise::ErrorIf(prtl_bc.size() > 1,
                       "invalid `grid.boundaries.particles`",
                       HERE);
        if (engine_enum == SimEngine::SRPIC) {
          raise::ErrorIf(flds_bc[0].size() != 2,
                         "invalid `grid.boundaries.fields`",
                         HERE);
          flds_bc_enum.push_back(
            { FldsBC::pick(fmt::toLower(flds_bc[0][0]).c_str()),
              FldsBC::pick(fmt::toLower(flds_bc[0][1]).c_str()) });
          flds_bc_enum.push_back({ FldsBC::AXIS, FldsBC::AXIS });
          if (dim == Dim::_3D) {
            flds_bc_enum.push_back({ FldsBC::PERIODIC, FldsBC::PERIODIC });
          }
          raise::ErrorIf(prtl_bc[0].size() != 2,
                         "invalid `grid.boundaries.particles`",
                         HERE);
          prtl_bc_enum.push_back(
            { PrtlBC::pick(fmt::toLower(prtl_bc[0][0]).c_str()),
              PrtlBC::pick(fmt::toLower(prtl_bc[0][1]).c_str()) });
          prtl_bc_enum.push_back({ PrtlBC::AXIS, PrtlBC::AXIS });
          if (dim == Dim::_3D) {
            prtl_bc_enum.push_back({ PrtlBC::PERIODIC, PrtlBC::PERIODIC });
          }
        } else {
          raise::ErrorIf(flds_bc[0].size() != 1,
                         "invalid `grid.boundaries.fields`",
                         HERE);
          raise::ErrorIf(prtl_bc[0].size() != 1,
                         "invalid `grid.boundaries.particles`",
                         HERE);
          flds_bc_enum.push_back(
            { FldsBC::HORIZON, FldsBC::pick(fmt::toLower(flds_bc[0][0]).c_str()) });
          flds_bc_enum.push_back({ FldsBC::AXIS, FldsBC::AXIS });
          if (dim == Dim::_3D) {
            flds_bc_enum.push_back({ FldsBC::PERIODIC, FldsBC::PERIODIC });
          }
          prtl_bc_enum.push_back(
            { PrtlBC::HORIZON, PrtlBC::pick(fmt::toLower(prtl_bc[0][0]).c_str()) });
          prtl_bc_enum.push_back({ PrtlBC::AXIS, PrtlBC::AXIS });
          if (dim == Dim::_3D) {
            prtl_bc_enum.push_back({ PrtlBC::PERIODIC, PrtlBC::PERIODIC });
          }
        }
      }

      raise::ErrorIf(flds_bc_enum.size() != (std::size_t)dim,
                     "invalid inferred `grid.boundaries.fields`",
                     HERE);
      raise::ErrorIf(prtl_bc_enum.size() != (std::size_t)dim,
                     "invalid inferred `grid.boundaries.particles`",
                     HERE);
      boundaries_t<FldsBC> flds_bc_pairwise;
      boundaries_t<PrtlBC> prtl_bc_pairwise;
      for (auto d { 0u }; d < (dim_t)dim; ++d) {
        raise::ErrorIf(
          flds_bc_enum[d].size() != 2,
          fmt::format("invalid inferred `grid.boundaries.fields[%d]`", d),
          HERE);
        raise::ErrorIf(
          prtl_bc_enum[d].size() != 2,
          fmt::format("invalid inferred `grid.boundaries.particles[%d]`", d),
          HERE);
        flds_bc_pairwise.push_back({ flds_bc_enum[d][0], flds_bc_enum[d][1] });
        prtl_bc_pairwise.push_back({ prtl_bc_enum[d][0], prtl_bc_enum[d][1] });
      }
      return { flds_bc_pairwise, prtl_bc_pairwise };
    }

    void Boundaries::read(Dimension                   dim,
                          const Coord&                coord_enum,
                          const boundaries_t<real_t>& extent_pairwise,
                          const toml::value&          toml_data) {
      if (needs_match_boundaries) {
        if (coord_enum == Coord::Cart) {
          auto min_extent = std::numeric_limits<real_t>::max();
          for (const auto& e : extent_pairwise) {
            min_extent = std::min(min_extent, e.second - e.first);
          }
          const auto default_ds = min_extent * defaults::bc::match::ds_frac;
          try {
            auto ds = toml::find<real_t>(toml_data, "grid", "boundaries", "match", "ds");
            for (auto d = 0u; d < dim; ++d) {
              match_ds_array.push_back({ ds, ds });
            }
          } catch (...) {
            try {
              const auto ds = toml::find<std::vector<std::vector<real_t>>>(
                toml_data,
                "grid",
                "boundaries",
                "match",
                "ds");
              raise::ErrorIf(ds.size() != dim,
                             "invalid # in `grid.boundaries.match.ds`",
                             HERE);
              for (auto d = 0u; d < dim; ++d) {
                if (ds[d].size() == 1) {
                  match_ds_array.push_back({ ds[d][0], ds[d][0] });
                } else if (ds[d].size() == 2) {
                  match_ds_array.push_back({ ds[d][0], ds[d][1] });
                } else if (ds[d].size() == 0) {
                  match_ds_array.push_back({});
                } else {
                  raise::Error("invalid `grid.boundaries.match.ds`", HERE);
                }
              }
            } catch (...) {
              for (auto d = 0u; d < dim; ++d) {
                match_ds_array.push_back({ default_ds, default_ds });
              }
            }
          }
        } else {
          auto r_extent = extent_pairwise[0].second - extent_pairwise[0].first;
          const auto ds = toml::find_or<real_t>(
            toml_data,
            "grid",
            "boundaries",
            "match",
            "ds",
            r_extent * defaults::bc::match::ds_frac);
          match_ds_array.push_back({ ds, ds });
        }
      }

      if (needs_absorb_boundaries) {
        if (coord_enum == Coord::Cart) {
          auto min_extent = std::numeric_limits<real_t>::max();
          for (const auto& e : extent_pairwise) {
            min_extent = std::min(min_extent, e.second - e.first);
          }
          absorb_ds = toml::find_or(toml_data,
                                    "grid",
                                    "boundaries",
                                    "absorb",
                                    "ds",
                                    min_extent * defaults::bc::absorb::ds_frac);
        } else {
          auto r_extent = extent_pairwise[0].second - extent_pairwise[0].first;
          absorb_ds     = toml::find_or(toml_data,
                                    "grid",
                                    "boundaries",
                                    "absorb",
                                    "ds",
                                    r_extent * defaults::bc::absorb::ds_frac);
        }
      }

      if (needs_atmosphere_boundaries) {
        atmosphere_temperature = toml::find<real_t>(toml_data,
                                                    "grid",
                                                    "boundaries",
                                                    "atmosphere",
                                                    "temperature");
        atmosphere_height      = toml::find<real_t>(toml_data,
                                               "grid",
                                               "boundaries",
                                               "atmosphere",
                                               "height");
        atmosphere_density     = toml::find<real_t>(toml_data,
                                                "grid",
                                                "boundaries",
                                                "atmosphere",
                                                "density");
        atmosphere_ds =
          toml::find_or(toml_data, "grid", "boundaries", "atmosphere", "ds", ZERO);
        atmosphere_g       = atmosphere_temperature / atmosphere_height;
        atmosphere_species = toml::find<std::pair<spidx_t, spidx_t>>(
          toml_data,
          "grid",
          "boundaries",
          "atmosphere",
          "species");
      }
    }

    void Boundaries::setParams(SimulationParams* params) const {
      if (needs_match_boundaries) {
        params->set("grid.boundaries.match.ds", match_ds_array);
      }
      if (needs_absorb_boundaries) {
        params->set("grid.boundaries.absorb.ds", absorb_ds);
      }
      if (needs_atmosphere_boundaries) {
        params->set("grid.boundaries.atmosphere.temperature",
                    atmosphere_temperature);
        params->set("grid.boundaries.atmosphere.density", atmosphere_density);
        params->set("grid.boundaries.atmosphere.height", atmosphere_height);
        params->set("grid.boundaries.atmosphere.ds", atmosphere_ds);
        params->set("grid.boundaries.atmosphere.g", atmosphere_g);
        params->set("grid.boundaries.atmosphere.species", atmosphere_species);
      }
    }

  } // namespace params
} // namespace ntt
