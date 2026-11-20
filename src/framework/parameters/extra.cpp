#include "framework/parameters/extra.h"

#include "defaults.h"
#include "global.h"

#include "utils/numeric.h"

#include "framework/parameters/extra.h"
#include "framework/parameters/parameters.h"

#include <toml11/toml.hpp>

#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace ntt {
  namespace params {

    void Extra::read(const std::map<std::string, bool>& extra,
                     const toml::value&                 toml_data,
                     const SimulationParams* const      params) {
      if (extra.at("synchrotron_drag")) {
        synchrotron_gamma_rad = toml::find_or(toml_data,
                                              "radiation",
                                              "drag",
                                              "synchrotron",
                                              "gamma_rad",
                                              defaults::synchrotron::gamma_rad);
      }

      if (extra.at("compton_drag")) {
        compton_gamma_rad = toml::find_or(toml_data,
                                          "radiation",
                                          "drag",
                                          "compton",
                                          "gamma_rad",
                                          defaults::compton::gamma_rad);
      }

      if (extra.at("synchrotron_emission")) {
        synchrotron_gamma_rad      = toml::find_or(toml_data,
                                              "radiation",
                                              "drag",
                                              "synchrotron",
                                              "gamma_rad",
                                              defaults::synchrotron::gamma_rad);
        synchrotron_gamma_qed      = toml::find_or(toml_data,
                                              "radiation",
                                              "emission",
                                              "synchrotron",
                                              "gamma_qed",
                                              defaults::synchrotron::gamma_qed);
        synchrotron_energy_min     = toml::find_or(toml_data,
                                               "radiation",
                                               "emission",
                                               "synchrotron",
                                               "photon_energy_min",
                                               defaults::synchrotron::energy_min);
        synchrotron_photon_weight  = toml::find_or(toml_data,
                                                  "radiation",
                                                  "emission",
                                                  "synchrotron",
                                                  "photon_weight",
                                                  ONE);
        synchrotron_photon_species = toml::find<spidx_t>(toml_data,
                                                         "radiation",
                                                         "emission",
                                                         "synchrotron",
                                                         "photon_species");
        synchrotron_nominal_probability =
          params->template get<real_t>("scales.omegaB0") *
          static_cast<real_t>(0.1) *
          params->template get<real_t>("algorithms.timestep.dt") *
          SQR(synchrotron_gamma_qed.value() / synchrotron_gamma_rad.value()) /
          synchrotron_photon_weight.value();
        synchrotron_nominal_photon_energy = ONE /
                                            SQR(synchrotron_gamma_qed.value());
      }

      if (extra.at("compton_emission")) {
        compton_gamma_rad      = toml::find_or(toml_data,
                                          "radiation",
                                          "drag",
                                          "compton",
                                          "gamma_rad",
                                          defaults::compton::gamma_rad);
        compton_gamma_qed      = toml::find_or(toml_data,
                                          "radiation",
                                          "emission",
                                          "compton",
                                          "gamma_qed",
                                          defaults::compton::gamma_qed);
        compton_energy_min     = toml::find_or(toml_data,
                                           "radiation",
                                           "emission",
                                           "compton",
                                           "photon_energy_min",
                                           defaults::compton::energy_min);
        compton_photon_weight  = toml::find_or(toml_data,
                                              "radiation",
                                              "emission",
                                              "compton",
                                              "photon_weight",
                                              ONE);
        compton_photon_species = toml::find<spidx_t>(toml_data,
                                                     "radiation",
                                                     "emission",
                                                     "compton",
                                                     "photon_species");
        compton_nominal_probability =
          params->template get<real_t>("scales.omegaB0") *
          static_cast<real_t>(0.1) *
          params->template get<real_t>("algorithms.timestep.dt") *
          SQR(compton_gamma_qed.value() / compton_gamma_rad.value()) /
          compton_photon_weight.value();
        compton_nominal_photon_energy = ONE / SQR(compton_gamma_qed.value());
      }

      if (extra.at("bethe_heitler_emission")) {
        const auto bh = toml::find(toml_data, "radiation", "emission", "bethe_heitler");
        bethe_heitler_electron_species = toml::find<spidx_t>(bh, "electron_species");
        bethe_heitler_positron_species = toml::find<spidx_t>(bh, "positron_species");
        bethe_heitler_optical_depth = toml::find<real_t>(bh, "optical_depth");
        bethe_heitler_mass_ratio    = toml::find_or(
          bh,
          "mass_ratio",
          defaults::bethe_heitler::mass_ratio);
        bethe_heitler_pair_weight = toml::find_or(
          bh,
          "pair_weight",
          defaults::bethe_heitler::pair_weight);
        bethe_heitler_max_probability = toml::find_or(
          bh,
          "max_probability",
          defaults::bethe_heitler::max_probability);
        bethe_heitler_max_loss_fraction = toml::find_or(
          bh,
          "max_loss_fraction",
          defaults::bethe_heitler::max_loss_fraction);
        bethe_heitler_gamma_max       = toml::find_or(bh,
                                                "gamma_max",
                                                defaults::bethe_heitler::gamma_max);
        bethe_heitler_photon_spectrum = fmt::toLower(
          toml::find_or(bh,
                        "photon_spectrum",
                        defaults::bethe_heitler::photon_spectrum));
        // the keys each spectrum needs are required, the others unused
        const auto& spectrum = bethe_heitler_photon_spectrum.value();
        if (spectrum == "mono") {
          bethe_heitler_photon_energy = toml::find<real_t>(bh, "photon_energy");
        } else if (spectrum == "powerlaw") {
          bethe_heitler_photon_energy_min = toml::find<real_t>(bh, "photon_energy_min");
          bethe_heitler_photon_energy_max = toml::find<real_t>(bh, "photon_energy_max");
          bethe_heitler_photon_index = toml::find<real_t>(bh, "photon_index");
        } else if (spectrum == "blackbody") {
          bethe_heitler_photon_temperature = toml::find<real_t>(
            bh,
            "photon_temperature");
          bethe_heitler_photon_energy_min = toml::find_or(bh,
                                                          "photon_energy_min",
                                                          ZERO);
          bethe_heitler_photon_energy_max = toml::find<real_t>(bh, "photon_energy_max");
        } else {
          raise::Error("radiation.emission.bethe_heitler.photon_spectrum must "
                       "be mono, powerlaw or blackbody",
                       HERE);
        }
        // comparisons with a NaN are false: it would pass the checks below
        for (const auto& value : { bethe_heitler_optical_depth,
                                   bethe_heitler_mass_ratio,
                                   bethe_heitler_pair_weight,
                                   bethe_heitler_max_probability,
                                   bethe_heitler_max_loss_fraction,
                                   bethe_heitler_gamma_max,
                                   bethe_heitler_photon_energy,
                                   bethe_heitler_photon_energy_min,
                                   bethe_heitler_photon_energy_max,
                                   bethe_heitler_photon_index,
                                   bethe_heitler_photon_temperature }) {
          raise::ErrorIf(value.has_value() and not std::isfinite(value.value()),
                         "Bethe-Heitler parameters must be finite",
                         HERE);
        }
        raise::ErrorIf(bethe_heitler_electron_species.value() ==
                         bethe_heitler_positron_species.value(),
                       "Bethe-Heitler electron_species and positron_species "
                       "must be different",
                       HERE);
        raise::ErrorIf(bethe_heitler_optical_depth.value() < ZERO,
                       "Bethe-Heitler optical_depth must be >= 0",
                       HERE);
        raise::ErrorIf(bethe_heitler_pair_weight.value() <= ZERO,
                       "Bethe-Heitler pair_weight must be > 0",
                       HERE);
        raise::ErrorIf(
          (bethe_heitler_pair_weight.value() != ONE) and
            (not params->template get<bool>("particles.use_weights")),
          "Bethe-Heitler pair_weight != 1 requires "
          "particles.use_weights = true",
          HERE);
        raise::ErrorIf((bethe_heitler_max_probability.value() <= ZERO) or
                         (bethe_heitler_max_probability.value() > ONE),
                       "Bethe-Heitler max_probability must be in (0, 1]",
                       HERE);
        raise::ErrorIf((bethe_heitler_max_loss_fraction.value() <= ZERO) or
                         (bethe_heitler_max_loss_fraction.value() >= ONE),
                       "Bethe-Heitler max_loss_fraction must be in (0, 1)",
                       HERE);
      }

      if (toml_data.contains("two_body") and
          toml_data.at("two_body").contains("thomson_mean_free_path")) {
        twobody_thomson_mean_free_path = toml::find<real_t>(
          toml_data,
          "two_body",
          "thomson_mean_free_path");
        twobody_thomson_optical_depth = ONE /
                                        twobody_thomson_mean_free_path.value();
      } else {
        twobody_thomson_optical_depth = toml::find_or<real_t>(
          toml_data,
          "two_body",
          "thomson_optical_depth",
          defaults::twobody::thomson_optical_depth);
        twobody_thomson_mean_free_path = ONE /
                                         twobody_thomson_optical_depth.value();
      }

      // find two-body interactions
      const auto twobody_tab = toml::find_or<toml::array>(toml_data,
                                                          "two_body",
                                                          "interaction",
                                                          toml::array {});
      for (const auto& tbint : twobody_tab) {
        twobody_interactions.push_back(TwoBodyInteractionParams {
          .type = TwoBodyInteraction::from_string(
            toml::find<std::string>(tbint, "type")),
          .group1    = toml::find<std::vector<spidx_t>>(tbint, "group1"),
          .group2    = toml::find_or<std::vector<spidx_t>>(tbint, "group2", {}),
          .interval  = toml::find_or<timestep_t>(tbint, "interval", 1),
          .tile_size = toml::find_or<ncells_t>(tbint, "tile_size", 4u),
          .recoil1   = toml::find_or<bool>(tbint, "recoil1", true),
          .recoil2   = toml::find_or<bool>(tbint, "recoil2", true) });
      }
    }

    void Extra::setParams(const std::map<std::string, bool>& extra,
                          SimulationParams*                  params) const {
      if (extra.at("synchrotron_drag")) {
        params->set("radiation.drag.synchrotron.gamma_rad",
                    synchrotron_gamma_rad.value());
      }

      if (extra.at("compton_drag")) {
        params->set("radiation.drag.compton.gamma_rad", compton_gamma_rad.value());
      }

      if (extra.at("synchrotron_emission")) {
        params->set("radiation.drag.synchrotron.gamma_rad",
                    synchrotron_gamma_rad.value());
        params->set("radiation.emission.synchrotron.gamma_qed",
                    synchrotron_gamma_qed.value());
        params->set("radiation.emission.synchrotron.photon_energy_min",
                    synchrotron_energy_min.value());
        params->set("radiation.emission.synchrotron.photon_weight",
                    synchrotron_photon_weight.value());
        params->set("radiation.emission.synchrotron.photon_species",
                    synchrotron_photon_species.value());
        params->set("radiation.emission.synchrotron.nominal_probability",
                    synchrotron_nominal_probability.value());
        params->set("radiation.emission.synchrotron.nominal_photon_energy",
                    synchrotron_nominal_photon_energy.value());
      }

      if (extra.at("compton_emission")) {
        params->set("radiation.drag.compton.gamma_rad", compton_gamma_rad.value());
        params->set("radiation.emission.compton.gamma_qed",
                    compton_gamma_qed.value());
        params->set("radiation.emission.compton.photon_energy_min",
                    compton_energy_min.value());
        params->set("radiation.emission.compton.photon_weight",
                    compton_photon_weight.value());
        params->set("radiation.emission.compton.photon_species",
                    compton_photon_species.value());
        params->set("radiation.emission.compton.nominal_probability",
                    compton_nominal_probability.value());
        params->set("radiation.emission.compton.nominal_photon_energy",
                    compton_nominal_photon_energy.value());
      }

      if (extra.at("bethe_heitler_emission")) {
        const std::string prefix = "radiation.emission.bethe_heitler.";
        params->set(prefix + "electron_species",
                    bethe_heitler_electron_species.value());
        params->set(prefix + "positron_species",
                    bethe_heitler_positron_species.value());
        params->set(prefix + "optical_depth", bethe_heitler_optical_depth.value());
        params->set(prefix + "mass_ratio", bethe_heitler_mass_ratio.value());
        params->set(prefix + "pair_weight", bethe_heitler_pair_weight.value());
        params->set(prefix + "max_probability",
                    bethe_heitler_max_probability.value());
        params->set(prefix + "max_loss_fraction",
                    bethe_heitler_max_loss_fraction.value());
        params->set(prefix + "gamma_max", bethe_heitler_gamma_max.value());
        params->set(prefix + "photon_spectrum",
                    bethe_heitler_photon_spectrum.value());
        params->set(prefix + "photon_energy",
                    bethe_heitler_photon_energy.value_or(ZERO));
        params->set(prefix + "photon_energy_min",
                    bethe_heitler_photon_energy_min.value_or(ZERO));
        params->set(prefix + "photon_energy_max",
                    bethe_heitler_photon_energy_max.value_or(ZERO));
        params->set(prefix + "photon_index",
                    bethe_heitler_photon_index.value_or(ZERO));
        params->set(prefix + "photon_temperature",
                    bethe_heitler_photon_temperature.value_or(ZERO));
      }

      params->set("two_body.thomson_optical_depth",
                  twobody_thomson_optical_depth.value());
      params->set("two_body.thomson_mean_free_path",
                  twobody_thomson_mean_free_path.value());
      params->set("two_body.interaction", twobody_interactions);
    }
  } // namespace params
} // namespace ntt
