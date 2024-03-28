#include "wrapper.h"

#include "sim_params.h"

#include "meshblock/particles.h"
#include "utilities/utils.h"

#include <toml.hpp>

#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ntt {

  SimulationParams::SimulationParams(const toml::value& inputdata) :
    m_raw_data { inputdata } {

    short            dim;
    SimulationEngine engine;
    /* Simulation ----------------------------------------------------------- */
    {
      this->set("simulation.name",
                toml::find<std::string>(m_raw_data, "simulation", "name"));
      this->set("simulation.runtime",
                toml::find<real_t>(m_raw_data, "simulation", "runtime"));
      const auto engine_str = toml::find<std::string>(m_raw_data,
                                                      "simulation",
                                                      "engine");
      if (ToLower(engine_str) == "srpic") {
        engine = SimulationEngine::SRPIC;
      } else if (ToLower(engine_str) == "grpic") {
        engine = SimulationEngine::GRPIC;
      } else {
        NTTHostError("unrecognized simulation engine");
      }
      this->set("simulation.engine", engine);
    }

    /* Grid ----------------------------------------------------------------- */
    {
      auto res = toml::find<std::vector<unsigned int>>(m_raw_data,
                                                       "grid",
                                                       "resolution");
      dim      = static_cast<short>(res.size());
      NTTHostErrorIf(res.size() > 3 || res.size() < 1,
                     "undefined dimension inferred from `grid.resolution`");
      this->set("grid.resolution", res);
      this->set("grid.dim", dim);

      auto metric = toml::find<std::string>(m_raw_data, "grid", "metric", "metric");
      std::string coord;
      if (metric == "minkowski") {
        coord = "cartesian";
      } else if (metric[0] == 'q') {
        coord = "qspherical";
      } else {
        coord = "spherical";
      }
      this->set("grid.metric.metric", metric);
      this->set("grid.metric.coord", coord);

      auto ext = toml::find<std::vector<std::vector<real_t>>>(m_raw_data,
                                                              "grid",
                                                              "extent");
      NTTHostErrorIf(ext.size() < 1, "not enough values in `grid.extent` input");
      if (coord != "cartesian") {
        NTTHostErrorIf(dim < 2, "not enough dimensions for spherical geometry");
        NTTHostErrorIf(ext.size() != 1, "theta and/or phi extent are unnecessary");
        ext.push_back({ ZERO, constant::PI });
        ext.push_back({ ZERO, constant::TWO_PI });
      }
      ext.erase(ext.begin() + dim, ext.end());
      for (auto& e : ext) {
        NTTHostErrorIf(e.size() != 2, "extent must be a pair of values [min, max]")
      }
      if (coord == "cartesian") {
        real_t dx { -ONE };
        short  d = 0;
        for (auto& e : ext) {
          if (dx == -ONE) {
            dx = (e[1] - e[0]) / res[d];
          } else {
            NTTHostErrorIf((dx != (e[1] - e[0]) / res[d]),
                           "dx != dy or dx != dz in cartesian geometry");
          }
          d++;
        }
      }
      this->set("grid.extent", ext);

      // special case for qspherical
      if (coord == "qspherical") {
        this->set("grid.metric.qsph_r0",
                  toml::find_or<real_t>(m_raw_data,
                                        "grid",
                                        "metric",
                                        "qsph_r0",
                                        defaults::qsph::r0));
        this->set("grid.metric.qsph_h",
                  toml::find_or<real_t>(m_raw_data,
                                        "grid",
                                        "metric",
                                        "qsph_h",
                                        defaults::qsph::h));
      }
      // special case for GR
      if (engine == SimulationEngine::GRPIC) {
        const auto ks_a  = toml::find_or<real_t>(m_raw_data,
                                                "grid",
                                                "metric",
                                                "ks_a",
                                                defaults::ks::a);
        const auto ks_rh = ONE + math::sqrt(ONE - SQR(ks_a));
        this->set("grid.metric.ks_a", ks_a);
        this->set("grid.metric.ks_rh", ks_rh);
      }

      // boundary conditions />
      auto has_atmosphere = false;
      {
        auto has_absorb = false;
        auto field_bcs  = toml::find<std::vector<std::vector<std::string>>>(
          m_raw_data,
          "grid",
          "boundaries",
          "fields");
        NTTHostErrorIf(field_bcs.size() < 1 || field_bcs.size() > 3,
                       "invalid field boundaries");
        for (const auto& fb : field_bcs) {
          NTTHostErrorIf((fb.size() > 2) || (fb.size() < 1),
                         "invalid field boundary");
        }
        if (coord != "cartesian") {
          NTTHostErrorIf(field_bcs.size() > 1,
                         "for spherical, only specify r-boundaries");
          field_bcs.push_back({ "axis" });
          field_bcs.push_back({ "periodic" });
        }
        if (engine == SimulationEngine::GRPIC) {
          NTTHostErrorIf((field_bcs[0].size() > 1) &&
                           (ToLower(field_bcs[0][0]) != "horizon"),
                         "In GR, rmin boundaries should always be horizon");
          if (field_bcs[0].size() == 1) {
            field_bcs[0].insert(field_bcs[0].begin(), "horizon");
          }
        }
        field_bcs.erase(field_bcs.begin() + dim, field_bcs.end());
        std::vector<std::vector<FieldsBC>> field_bcs_enum;
        for (const auto& fb : field_bcs) {
          std::vector<FieldsBC> fb_enum;
          for (const auto& fb_xi : fb) {
            const auto bc = pickFrom<ParticleBC>(pb_xi);
            fb_enum.push_back(bc);
            NTTHostErrorIf(bc == FieldsBC::NONE, "unrecognized field boundary");
            NTTHostErrorIf((engine != SimulationEngine::GRPIC) &&
                             (bc == FieldsBC::HORIZON),
                           "Horizon boundaries are only applicable for GR");
            has_absorb     = has_absorb || (bc == FieldsBC::ABSORB);
            has_atmosphere = has_atmosphere || (bc == FieldsBC::ATMOSPHERE);
            if (fb_enum.empty()) {
              NTTHostError("unrecognized field boundary");
            } else if (fb_enum.size() == 1) {
              // duplicate boundaries if only one specified
              fb_enum.push_back(fb_enum[0]);
            }
          }
          field_bcs_enum.push_back(fb_enum);
        }
        this->set("grid.boundaries.fields", field_bcs_enum);
      }

      {
        auto prtl_bcs = toml::find<std::vector<std::vector<std::string>>>(
          m_raw_data,
          "grid",
          "boundaries",
          "particles");
        NTTHostErrorIf(prtl_bcs.size() < 1 || prtl_bcs.size() > 3,
                       "invalid prtl boundaries");
        for (const auto& pb : prtl_bcs) {
          NTTHostErrorIf((pb.size() > 2) || (pb.size() < 1),
                         "invalid prtl boundary");
        }
        if (coord != "cartesian") {
          NTTHostErrorIf(prtl_bcs.size() > 1,
                         "for spherical, only specify r-boundaries");
          prtl_bcs.push_back({ "axis" });
          prtl_bcs.push_back({ "periodic" });
        }
        if (engine == SimulationEngine::GRPIC) {
          NTTHostErrorIf((prtl_bcs[0].size() > 1) &&
                           (ToLower(prtl_bcs[0][0]) != "horizon"),
                         "In GR, rmin boundaries should always be horizon");
          if (prtl_bcs[0].size() == 1) {
            prtl_bcs[0].insert(prtl_bcs[0].begin(), "horizon");
          }
        }
        prtl_bcs.erase(prtl_bcs.begin() + dim, prtl_bcs.end());
        std::vector<std::vector<ParticleBC>> prtl_bcs_enum;
        for (const auto& pb : prtl_bcs) {
          std::vector<ParticleBC> pb_enum;
          for (const auto& pb_xi : pb) {
            const auto bc = pickFrom<ParticleBC>(pb_xi);
            pb_enum.push_back(bc);
            has_absorb     = has_absorb || (bc == ParticleBC::ABSORB);
            has_atmosphere = has_atmosphere || (bc == ParticleBC::ATMOSPHERE);
            NTTHostErrorIf(bc == ParticleBC::NONE,
                           "unrecognized particle boundary");
            NTTHostErrorIf((engine != SimulationEngine::GRPIC) &&
                             (bc == ParticleBC::HORIZON),
                           "Horizon boundaries are only applicable for GR");
            if (pb_enum.empty()) {
              NTTHostError("unrecognized field boundary");
            } else if (pb_enum.size() == 1) {
              // duplicate boundaries if only one specified
              pb_enum.push_back(pb_enum[0]);
            }
          }
          prtl_bcs_enum.push_back(pb_enum);
        }
        this->set("grid.boundaries.particles", prtl_bcs_enum);
      }

      if (has_absorb) {
        real_t d_absorb_default = (ext[0][1] - ext[0][0]) *
                                  defaults::bc::d_absorb_frac;
        if (coord == "cartesian") {
          for (const auto& e : ext) {
            const auto ds = (e[1] - e[0]) * defaults::bc::d_absorb_frac;
            if (ds < d_absorb_default) {
              d_absorb_default = ds;
            }
          }
        }
        this->set("grid.boundaries.absorb_d",
                  toml::find_or<real_t>(m_raw_data,
                                        "grid",
                                        "boundaries",
                                        "d_absorb",
                                        d_absorb_default));
        const auto absorb_coeff = toml::find_or<real_t>(m_raw_data,
                                                        "grid",
                                                        "boundaries",
                                                        "absorb_coeff",
                                                        defaults::bc::absorb_coeff);
        NTTHostErrorIf(absorb_coeff == ZERO, "absorb_coeff must be non-zero");
        this->set("grid.boundaries.absorb_coeff", absorb_coeff);
      }
      NTTHostErrorIf(has_atmosphere && (engine == SimulationEngine::GRPIC),
                     "GRPIC does not support atmosphere boundary conditions");
    }
    // </ boundary conditions

    /* Scales --------------------------------------------------------------- */
    {
      const auto larmor0 = toml::find<real_t>(m_raw_data, "scales", "larmor0");
      const auto skindepth0 = toml::find<real_t>(m_raw_data, "scales", "skindepth0");
      NTTHostErrorIf((larmor0 <= ZERO0) || (skindepth0 <= ZERO),
                     "larmor0 and skindepth0 must be positive");

      this->set("scales.larmor0", larmor0);
      this->set("scales.skindepth0", skindepth0);
      this->promiseToDefine("scales.V0");
      this->promiseToDefine("scales.n0");
      this->promiseToDefine("scales.q0");
      this->set("scales.sigma0", SQR(skindepth0 / larmor0));
      this->set("scales.B0", ONE / larmor0);
      this->set("scales.omegaB0", ONE / larmor0);
    }

    /* Algorithms ----------------------------------------------------------- */
    {
      this->set("algorithms.current_filters",
                toml::find<unsigned short>(m_raw_data,
                                           "algorithms",
                                           "current_filters",
                                           defaults::bc::current_filters));
      // toggles
      this->set(
        "algorithms.toggles.fieldsolver",
        toml::find_or<bool>(m_raw_data, "algorithms", "toggles", "fieldsolver", true));
      this->set(
        "algorithms.toggles.deposit",
        toml::find_or<bool>(m_raw_data, "algorithms", "toggles", "deposit", true));
      const auto extforce = toml::find_or<bool>(m_raw_data,
                                                "algorithms",
                                                "toggles",
                                                "extforce",
                                                has_atmosphere);
      NTTHostErrorIf((engine == SimulationEngine::GRPIC) && extforce,
                     "extforce is not supported in GRPIC");
      this->set("algorithms.toggles.extforce", extforce);

      // timestep & correction
      this->set("algorithms.timestep.CFL",
                toml::find_or<real_t>(m_raw_data,
                                      "algorithms",
                                      "timestep",
                                      "CFL",
                                      defaults::cfl));
      this->set("algorithms.timestep.correction",
                toml::find_or<std::string>(m_raw_data,
                                           "algorithms",
                                           "timestep",
                                           "correction",
                                           defaults::correction));

      if (engine == SimulationEngine::GRPIC) {
        this->set("algorithms.gr.pusher_eps",
                  toml::find_or<real_t>(m_raw_data,
                                        "algorithms",
                                        "gr",
                                        "pusher_eps",
                                        defaults::gr::pusher_eps));
        this->set("algorithms.gr.pusher_niter",
                  toml::find_or<int>(m_raw_data,
                                     "algorithms",
                                     "gr",
                                     "pusher_niter",
                                     defaults::gr::pusher_niter));
      }
    }

    /* Particles ------------------------------------------------------------ */
    {
      const auto ppc0 = toml::find<real_t>(m_raw_data, "particles", "ppc0");
      this->set("particles.ppc0", ppc0);
      this->set("particles.use_weights",
                toml::find_or<bool>(m_raw_data, "particles", "use_weights", false));
      this->set("particles.sort_interval",
                toml::find_or<unsigned int>(m_raw_data,
                                            "particles",
                                            "sort_interval",
                                            defaults::sort_interval));
      std::vector<ParticleSpecies> species;
      const auto species_tab = toml::find<std::vector<toml::table>>(m_raw_data,
                                                                    "particles",
                                                                    "species");
      this->set("particles.nspec", species_tab.size());
      int idx = 1;
      for (const auto& sp : species_tab) {
        const auto label  = toml::find_or<std::string>(sp,
                                                      "label",
                                                      "s" + std::to_string(idx));
        const auto mass   = toml::find<real_t>(sp, "mass");
        const auto charge = toml::find<real_t>(sp, "charge");
        NTTHostErrorIf((charge != ZERO) && (mass == ZERO),
                       "mass of the charged species must be non-zero");
        const auto is_massless = (mass == ZERO) && (charge == ZERO);
        const auto def_pusher  = (is_massless ? defaults::ph_pusher
                                              : defaults::em_pusher);
        const auto maxnpart    = toml::find<std::size_t>(sp, "maxnpart");
        const auto pusher = toml::find_or<std::string>(sp, "pusher", def_pusher);
        const auto npayloads = toml::find_or<unsigned short>(sp, "n_payloads", 0);
        const auto cooling = toml::find_or<std::string>(sp, "cooling", "None");
        NTTHostErrorIf((ToLower(cooling) != "none") && is_massless,
                       "cooling is only applicable to massive particles");
        NTTHostErrorIf(
          (ToLower(pusher) == "photon") && !is_massless,
          "photon pusher is only applicable to massless particles");
        const auto pusher_enum  = pickFrom<ParticlePusher>(pusher);
        const auto cooling_enum = pickFrom<Cooling>(cooling);

        species.emplace_back(ParticleSpecies(idx,
                                             label,
                                             mass,
                                             charge,
                                             maxnpart,
                                             pusher_enum,
                                             cooling_enum,
                                             npayloads));

        this->set("particles.species", species);
        idx += 1;
      }
    }

    /* Output --------------------------------------------------------------- */
    {
      const auto fields = toml::find<std::vector<std::string>>(m_raw_data,
                                                               "output",
                                                               "fields",
                                                               {});
    }

    // this->set("grid.resolution",

  } // SimulationParams::SimulationParams

  //     m_correction         = get("algorithm", "correction", defaults::correction);
  //     m_enable_fieldsolver = get("algorithm", "fieldsolver_ON", true);
  //     m_enable_deposit     = get("algorithm", "deposit_ON", true);
  //     m_enable_extforce    = get("algorithm", "extforce_ON", false);

  //     /* ---------------------------------------------------------------------- */
  //     /* Particle parameters */
  //     /* ---------------------------------------------------------------------- */
  //     auto nspec = get("particles", "n_species", defaults::n_species);
  //     for (int i { 0 }; i < nspec; ++i) {
  //       auto label     = get("species_" + std::to_string(i + 1),
  //                        "label",
  //                        "s" + std::to_string(i + 1));
  //       auto mass      = get<float>("species_" + std::to_string(i + 1), "mass");
  //       auto charge    = get<float>("species_" + std::to_string(i + 1), "charge");
  //       auto npayloads = get<unsigned short>("species_" + std::to_string(i + 1),
  //                                            "n_payloads",
  //                                            0);
  //       auto maxnpart  = (std::size_t)(
  //         get<double>("species_" + std::to_string(i + 1), "maxnpart"));

  //       auto default_pusher = (mass == 0.0) && (charge == 0.0)
  //                               ? defaults::ph_pusher
  //                               : defaults::em_pusher;

  //       auto pusher_str = get("species_" + std::to_string(i + 1),
  //                             "pusher",
  //                             default_pusher,
  //                             options::pushers);

  //       auto cooling_str = get("species_" + std::to_string(i + 1),
  //                              "cooling",
  //                              (std::string)("None"),
  //                              options::cooling);

  //       ParticlePusher pusher { ParticlePusher::UNDEFINED };
  //       for (auto p : PusherIterator()) {
  //         if (stringizeParticlePusher(p) == pusher_str) {
  //           pusher = p;
  //           break;
  //         }
  //       }
  //       Cooling cooling { Cooling::UNDEFINED };
  //       for (auto c : CoolingIterator()) {
  //         if (stringizeCooling(c) == cooling_str) {
  //           cooling = c;
  //           break;
  //         }
  //       }
  //       if (pusher == ParticlePusher::UNDEFINED) {
  //         NTTHostError("unrecognized pusher");
  //       }
  //       if (cooling == Cooling::UNDEFINED) {
  //         NTTHostError("unrecognized cooling mechanism");
  //       }
  //       m_species.emplace_back(
  //         ParticleSpecies(i + 1, label, mass, charge, maxnpart, pusher, cooling, npayloads));
  //     }
  //     m_use_weights = get("particles", "use_weights", defaults::use_weights);

  //     /* ---------------------------------------------------------------------- */
  //     /* Identifying the metric & coordinate system */
  //     /* ---------------------------------------------------------------------- */
  //     m_metric = SIMULATION_METRIC;
  //     if (m_metric == "minkowski") {
  //       m_coordinates = "cartesian";
  //     } else if (m_metric[0] == 'q') {
  //       m_coordinates = "qspherical";
  //     } else {
  //       m_coordinates = "spherical";
  //     }

  //     /* ---------------------------------------------------------------------- */
  //     /* Domain/coordinate specific */
  //     /* ---------------------------------------------------------------------- */
  //     m_domaindecomposition = get<std::vector<unsigned int>>(
  //       "domain",
  //       "decomposition",
  //       std::vector<unsigned int>());
  //     } else if (m_coordinates.find("spherical") != std::string::npos) {
  //       /* spherical/quasi-spherical
  //       ------------------------------------------ */ NTTHostErrorIf((m_extent.size()
  //       < 2), "not enough values in `extent` input"); m_extent.erase(m_extent.begin()
  //       + 2, m_extent.end()); if (m_coordinates == "qspherical") {
  //         m_metric_parameters[0] = get<real_t>("domain", "qsph_r0", ZERO);
  //         m_metric_parameters[1] = get<real_t>("domain", "qsph_h", ZERO);
  //       }
  //       m_metric_parameters[2] = get("domain",
  //                                    "sph_rabsorb",
  //                                    (real_t)(m_extent[1] * 0.9));
  //       m_metric_parameters[3] = get("domain", "absorb_coeff", ONE);

  //       // GR specific
  //       if (m_metric.find("kerr_schild") != std::string::npos) {
  //         const real_t spin      = get("domain", "a", ZERO);
  //         const real_t rh        = ONE + math::sqrt(ONE - spin * spin);
  //         m_metric_parameters[4] = spin;
  //         m_metric_parameters[5] = rh;

  //         m_gr_pusher_epsilon = get("algorithm", "gr_pusher_epsilon", (real_t)(1.0e-6));
  //         m_gr_pusher_niter = get("algorithm", "gr_pusher_niter", 10);
  //       }

  //       m_extent.push_back(ZERO);
  //       m_extent.push_back(constant::PI);
  //       m_extent.push_back(ZERO);
  //       m_extent.push_back(constant::TWO_PI);
  //     } else {
  //       NTTHostError("unrecognized coordinates: " + m_coordinates);
  //     }
  //     // leave only necessary extent/resolution (<= DIM)
  //     m_extent.erase(m_extent.begin() + 2 * (short)(dim), m_extent.end());
  //     m_resolution.erase(m_resolution.begin() + (short)(dim), m_resolution.end());

  //     auto  boundaries = get<std::vector<std::vector<std::string>>>("domain",
  //                                                                  "boundaries");
  //     short b { 0 };
  //     for (auto& bc_xi : boundaries) {
  //       std::vector<BoundaryCondition> boundaries_xi;
  //       for (auto& bc : bc_xi) {
  //         TestValidOption(bc, options::boundaries);
  //         if (bc == "PERIODIC") {
  //           boundaries_xi.push_back(BoundaryCondition::PERIODIC);
  //         } else if (bc == "ABSORB") {
  //           boundaries_xi.push_back(BoundaryCondition::ABSORB);
  //         } else if (bc == "OPEN") {
  //           boundaries_xi.push_back(BoundaryCondition::OPEN);
  //         } else if (bc == "CUSTOM") {
  //           boundaries_xi.push_back(BoundaryCondition::CUSTOM);
  //         } else if (bc == "AXIS") {
  //           boundaries_xi.push_back(BoundaryCondition::AXIS);
  //         } else {
  //           boundaries_xi.push_back(BoundaryCondition::UNDEFINED);
  //         }
  //       }
  //       m_boundaries.push_back(boundaries_xi);
  //       ++b;
  //       if (b >= (short)(dim)) {
  //         break;
  //       }
  //     }

  //     // fundamental parameters
  //     m_ppc0       = get<real_t>("units", "ppc0");
  //     m_larmor0    = get<real_t>("units", "larmor0");
  //     m_skindepth0 = get<real_t>("units", "skindepth0");
  //     m_V0         = -ONE; // defined later

  //     // if dt not specified (== -1), will use CFL to calculate it
  //     m_dt  = get("algorithm", "dt", -ONE);
  //     m_cfl = get("algorithm", "CFL", defaults::cfl);
  //     assert(m_cfl > 0);

  //     // number of current filter passes
  //     m_current_filters = get("algorithm", "current_filters", defaults::current_filters);

  //     m_sort_interval = get("particles", "sort_interval", 100);
  // #if defined(MPI_ENABLED)
  //     // sort every step when MPI is enabled
  //     m_sort_interval = 1;
  // #endif

  //     // output params
  //     m_output_format   = get("output",
  //                           "format",
  //                           defaults::output::format,
  //                           options::outputs);
  //     m_output_interval = get("output", "interval", defaults::output::interval);
  //     m_output_interval_time = get("output", "interval_time", -1.0);
  //     m_output_fields        = get<std::vector<std::string>>("output",
  //                                                     "fields",
  //                                                     std::vector<std::string>());
  //     m_output_particles     = get<std::vector<std::string>>("output",
  //                                                        "particles",
  //                                                        std::vector<std::string>());
  //     m_output_mom_smooth = get("output", "mom_smooth", defaults::output::mom_smooth);
  //     m_output_prtl_stride = get("output", "prtl_stride", defaults::output::prtl_stride);
  //     m_output_as_is  = get("output", "as_is", false);
  //     m_output_ghosts = get("output", "ghosts", false);

  //     // diagnostic params
  //     m_diag_interval   = get("diagnostics", "interval",
  //     defaults::diag_interval); m_blocking_timers = get("diagnostics",
  //                             "blocking_timers",
  //                             defaults::blocking_timers);

  //     // algorithm specific parameters
  //     // ... GCA
  //     m_gca_EovrB_max        = get("GCA", "EovrB_max", defaults::gca::EovrB_max);
  //     m_gca_larmor_max       = get("GCA", "larmor_max", ZERO);
  //     // ... Synchrotron
  //     m_synchrotron_gammarad = get("synchrotron", "gamma_rad", -ONE);
  //   }
} // namespace ntt