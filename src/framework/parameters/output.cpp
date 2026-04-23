#include "framework/parameters/output.h"

#include "defaults.h"
#include "global.h"

#include "utils/error.h"
#include "utils/log.h"

#include <toml11/toml.hpp>

namespace ntt {
  namespace params {

    void Output::read(Dimension dim, std::size_t nspec, const toml::value& toml_data) {
      format = toml::find_or(toml_data, "output", "format", defaults::output::format);
      global_interval      = toml::find_or(toml_data,
                                      "output",
                                      "interval",
                                      defaults::output::interval);
      global_interval_time = toml::find_or<simtime_t>(toml_data,
                                                      "output",
                                                      "interval_time",
                                                      -1.0);
      raise::ErrorIf(
        not toml::find_or<bool>(toml_data, "output", "separate_files", true),
        "separate_files=false is deprecated",
        HERE);

      categories.emplace();
      for (const auto& category : { "fields", "particles", "spectra", "stats" }) {
        const auto q_int               = toml::find_or<timestep_t>(toml_data,
                                                     "output",
                                                     category,
                                                     "interval",
                                                     0);
        const auto q_int_time          = toml::find_or<simtime_t>(toml_data,
                                                         "output",
                                                         category,
                                                         "interval_time",
                                                         -1.0);
        (*categories)[category].enable = toml::find_or(toml_data,
                                                       "output",
                                                       category,
                                                       "enable",
                                                       true);
        if ((q_int == 0) and (q_int_time == -1.0)) {
          (*categories)[category].interval      = global_interval.value();
          (*categories)[category].interval_time = global_interval_time.value();
        } else {
          (*categories)[category].interval      = q_int;
          (*categories)[category].interval_time = q_int_time;
        }
      }

      /* Fields --------------------------------------------------------------- */
      const auto flds_out        = toml::find_or(toml_data,
                                          "output",
                                          "fields",
                                          "quantities",
                                          std::vector<std::string> {});
      const auto custom_flds_out = toml::find_or(toml_data,
                                                 "output",
                                                 "fields",
                                                 "custom",
                                                 std::vector<std::string> {});
      if (flds_out.size() == 0) {
        raise::Warning("No fields output specified", HERE);
      }
      fields_quantities        = flds_out;
      fields_custom_quantities = custom_flds_out;
      fields_mom_smooth        = toml::find_or(toml_data,
                                        "output",
                                        "fields",
                                        "mom_smooth",
                                        defaults::output::mom_smooth);
      fields_downsampling.emplace();
      try {
        auto field_dwn_ = toml::find<std::vector<unsigned int>>(toml_data,
                                                                "output",
                                                                "fields",
                                                                "downsampling");
        for (const auto& dwn : field_dwn_) {
          fields_downsampling->push_back(dwn);
        }
      } catch (...) {
        try {
          auto field_dwn_ = toml::find<unsigned int>(toml_data,
                                                     "output",
                                                     "fields",
                                                     "downsampling");
          for (auto i = 0u; i < dim; ++i) {
            fields_downsampling->push_back(field_dwn_);
          }
        } catch (...) {
          for (auto i = 0u; i < dim; ++i) {
            fields_downsampling->push_back(1u);
          }
        }
      }
      raise::ErrorIf(fields_downsampling->size() > 3,
                     "invalid `output.fields.downsampling`",
                     HERE);
      if (fields_downsampling->size() > dim) {
        fields_downsampling->erase(fields_downsampling->begin() + (std::size_t)(dim),
                                   fields_downsampling->end());
      }
      for (const auto& dwn : fields_downsampling.value()) {
        raise::ErrorIf(dwn == 0, "downsampling factor must be nonzero", HERE);
      }

      /* Particles ------------------------------------------------------------ */
      auto all_specs = std::vector<spidx_t> {};
      for (auto i = 0u; i < nspec; ++i) {
        all_specs.push_back(static_cast<spidx_t>(i + 1));
      }
      particles_species = toml::find_or(toml_data,
                                        "output",
                                        "particles",
                                        "species",
                                        all_specs);
      particles_stride  = toml::find_or(toml_data,
                                       "output",
                                       "particles",
                                       "stride",
                                       defaults::output::prtl_stride);

      /* Spectra -------------------------------------------------------------- */
      spectra_e_min    = toml::find_or(toml_data,
                                    "output",
                                    "spectra",
                                    "e_min",
                                    defaults::output::spec_emin);
      spectra_e_max    = toml::find_or(toml_data,
                                    "output",
                                    "spectra",
                                    "e_max",
                                    defaults::output::spec_emax);
      spectra_log_bins = toml::find_or(toml_data,
                                       "output",
                                       "spectra",
                                       "log_bins",
                                       defaults::output::spec_log);
      spectra_n_bins   = toml::find_or(toml_data,
                                     "output",
                                     "spectra",
                                     "n_bins",
                                     defaults::output::spec_nbins);

      /* Stats ---------------------------------------------------------------- */
      stats_quantities        = toml::find_or(toml_data,
                                       "output",
                                       "stats",
                                       "quantities",
                                       defaults::output::stats_quantities);
      stats_custom_quantities = toml::find_or(toml_data,
                                              "output",
                                              "stats",
                                              "custom",
                                              std::vector<std::string> {});

      /* Debug ---------------------------------------------------------------- */
      debug_as_is = toml::find_or(toml_data, "output", "debug", "as_is", false);
      debug_ghosts = toml::find_or(toml_data, "output", "debug", "ghosts", false);
      if (debug_ghosts.value()) {
        for (const auto& dwn : fields_downsampling.value()) {
          raise::ErrorIf(
            dwn != 1,
            "full resolution required when outputting with ghost cells",
            HERE);
        }
      }
    }

    void Output::setParams(SimulationParams* params) const {
      params->set("output.format", format.value());
      params->set("output.interval", global_interval.value());
      params->set("output.interval_time", global_interval_time.value());
      for (const auto& [category, cat_params] : categories.value()) {
        params->set("output." + category + ".enable", cat_params.enable);
        params->set("output." + category + ".interval", cat_params.interval);
        params->set("output." + category + ".interval_time",
                    cat_params.interval_time);
      }

      params->set("output.fields.quantities", fields_quantities.value());
      params->set("output.fields.custom", fields_custom_quantities.value());
      params->set("output.fields.mom_smooth", fields_mom_smooth.value());
      params->set("output.fields.downsampling", fields_downsampling.value());

      params->set("output.particles.species", particles_species.value());
      params->set("output.particles.stride", particles_stride.value());

      params->set("output.spectra.e_min", spectra_e_min.value());
      params->set("output.spectra.e_max", spectra_e_max.value());
      params->set("output.spectra.log_bins", spectra_log_bins.value());
      params->set("output.spectra.n_bins", spectra_n_bins.value());

      params->set("output.stats.quantities", stats_quantities.value());
      params->set("output.stats.custom", stats_custom_quantities.value());

      params->set("output.debug.as_is", debug_as_is.value());
      params->set("output.debug.ghosts", debug_ghosts.value());
    }

  } // namespace params
} // namespace ntt
