#include "framework/parameters/parameters.h"

#include "defaults.h"
#include "enums.h"
#include "global.h"

#include "arch/mpi_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/numeric.h"

#include "framework/containers/species.h"
#include "framework/parameters/algorithms.h"
#include "framework/parameters/extra.h"
#include "framework/parameters/grid.h"
#include "framework/parameters/output.h"
#include "framework/parameters/particles.h"

#include <toml11/toml.hpp>

#include <cstddef>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ntt {

  /*
   * . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
   * Parameters that must not be changed during the checkpoint restart
   * . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
   */
  void SimulationParams::setImmutableParams(const toml::value& toml_data) {
    /* [simulation] --------------------------------------------------------- */
    const auto engine_enum = SimEngine::pick(
      fmt::toLower(toml::find<std::string>(toml_data, "simulation", "engine")).c_str());
    set("simulation.engine", engine_enum);

    /* grid and decomposition ------------------------------------------------ */
    params::Grid grid_params {};
    grid_params.read(engine_enum, toml_data);
    grid_params.setParams(this);

    /* [scales] ------------------------------------------------------------- */
    const auto larmor0 = toml::find<real_t>(toml_data, "scales", "larmor0");
    const auto skindepth0 = toml::find<real_t>(toml_data, "scales", "skindepth0");
    raise::ErrorIf(larmor0 <= ZERO || skindepth0 <= ZERO,
                   "larmor0 and skindepth0 must be positive",
                   HERE);
    set("scales.larmor0", larmor0);
    set("scales.skindepth0", skindepth0);
    set("scales.sigma0", SQR(skindepth0 / larmor0));
    set("scales.B0", ONE / larmor0);
    set("scales.omegaB0", ONE / larmor0);
    if (engine_enum == SimEngine::HYBRID) {
      /**
       * hybrid-engine parameters (read from the [hybrid] table; ignored by the
       * SR/GR engines):
       *   gamma_ad - adiabatic index of the massless electron fluid (1 = isothermal)
       *   theta0   - electron temperature T_e (code units); 0 = cold electrons
       *   dens_min - vacuum threshold for the Ohm's law; below it E ramps to 0 (units of n0)
       *   v_max    - characteristic flow speed for the hybrid CFL (code units)
       */
      set("hybrid.gamma_ad",
          toml::find_or<real_t>(toml_data,
                                "hybrid",
                                "gamma_ad",
                                defaults::hybrid::gamma_ad));
      set("hybrid.theta0",
          toml::find_or<real_t>(toml_data, "hybrid", "theta0", ZERO));
      // vacuum threshold for the hybrid Ohm's law. Above it E follows the usual
      // [...]/N; below it E ramps continuously to zero so that plasma-free cells
      // keep B frozen instead of amplifying the right-hand side by 1/dens_min.
      set("hybrid.dens_min",
          toml::find_or<real_t>(toml_data,
                                "hybrid",
                                "dens_min",
                                defaults::hybrid::dens_min));
      // optional user-set characteristic flow speed for the hybrid CFL (code
      // units); 0 -> dt set purely by the Alfven + whistler signal speeds.
      set("hybrid.v_max",
          toml::find_or<real_t>(toml_data, "hybrid", "v_max", ZERO));
    }

    /* [particles] ---------------------------------------------------------- */
    const auto ppc0 = toml::find<real_t>(toml_data, "particles", "ppc0");
    set("particles.ppc0", ppc0);
    raise::ErrorIf(ppc0 <= 0.0, "ppc0 must be positive", HERE);
    set("particles.use_weights",
        toml::find_or(toml_data, "particles", "use_weights", false));
    const auto global_clearing_interval = toml::find_or<timestep_t>(
      toml_data,
      "particles",
      "clear_interval",
      defaults::clear_interval);
    set("particles.clear_interval", global_clearing_interval);
    const auto global_spatial_sorting_interval = toml::find_or<timestep_t>(
      toml_data,
      "particles",
      "spatial_sorting_interval",
      0u);
    set("particles.spatial_sorting_interval", global_spatial_sorting_interval);

    set("scales.n0", ppc0 / get<real_t>("scales.V0"));
    set("scales.q0", get<real_t>("scales.V0") / (ppc0 * SQR(skindepth0)));

    /* [particles.species] -------------------------------------------------- */
    std::vector<ParticleSpecies> species;
    const auto species_tab = toml::find_or<toml::array>(toml_data,
                                                        "particles",
                                                        "species",
                                                        toml::array {});
    set("particles.nspec", species_tab.size());

    spidx_t idx = 1;
    for (const auto& sp : species_tab) {
      species.emplace_back(
        params::GetParticleSpecies(this,
                                   engine_enum,
                                   idx,
                                   sp,
                                   global_clearing_interval,
                                   global_spatial_sorting_interval));
      idx += 1;
    }
    set("particles.species", species);
  }

  /*
   * . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
   * Parameters that may be changed during the checkpoint restart
   * . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
   */
  void SimulationParams::setMutableParams(const toml::value& toml_data) {
    const auto engine_enum     = get<SimEngine>("simulation.engine");
    const auto coord_enum      = get<Coord>("grid.metric.coord");
    const auto dim             = get<Dimension>("grid.dim");
    const auto extent_pairwise = get<boundaries_t<real_t>>("grid.extent");

    /* [simulation] --------------------------------------------------------- */
    set("simulation.name",
        toml::find<std::string>(toml_data, "simulation", "name"));
    set("simulation.runtime",
        toml::find<simtime_t>(toml_data, "simulation", "runtime"));

    /* [grid.boundaraies] --------------------------------------------------- */
    const auto [flds_bc, prtl_bc] = params::GetBoundaryConditions(this,
                                                                  engine_enum,
                                                                  dim,
                                                                  coord_enum,
                                                                  toml_data);
    set("grid.boundaries.fields", flds_bc);
    set("grid.boundaries.particles", prtl_bc);

    /* [particles] ---------------------------------------------------------- */
    const auto species_tab               = toml::find_or<toml::array>(toml_data,
                                                        "particles",
                                                        "species",
                                                        toml::array {});
    std::vector<ParticleSpecies> species = get<std::vector<ParticleSpecies>>(
      "particles.species");
    raise::ErrorIf(species_tab.size() != species.size(),
                   "number of species changed after restart",
                   HERE);

    std::vector<ParticleSpecies> new_species;

    spidx_t idxM1 = 0;
    for (const auto& sp : species_tab) {
      const auto  maxnpart_real    = toml::find<double>(sp, "maxnpart");
      const auto  maxnpart         = static_cast<npart_t>(maxnpart_real);
      const auto& particle_species = species[idxM1];
      new_species.emplace_back(particle_species.index(),
                               particle_species.label(),
                               particle_species.mass(),
                               particle_species.charge(),
                               maxnpart,
                               particle_species.clearing_interval(),
                               particle_species.spatial_sorting_interval(),
                               particle_species.pusher(),
                               particle_species.use_tracking(),
                               particle_species.radiative_drag_flags(),
                               particle_species.emission_policy_flag(),
                               particle_species.npld_r(),
                               particle_species.npld_i());
      idxM1++;
    }
    set("particles.species", new_species);

    /* [output] ------------------------------------------------------------- */
    params::Output output_params;
    output_params.read(dim, get<std::size_t>("particles.nspec"), toml_data);
    output_params.setParams(this);

    /* [checkpoint] --------------------------------------------------------- */
    set("checkpoint.interval",
        toml::find_or(toml_data,
                      "checkpoint",
                      "interval",
                      defaults::checkpoint::interval));
    set("checkpoint.interval_time",
        toml::find_or<simtime_t>(toml_data, "checkpoint", "interval_time", -1.0));
    set("checkpoint.keep",
        toml::find_or(toml_data, "checkpoint", "keep", defaults::checkpoint::keep));
    auto walltime_str = toml::find_or(toml_data,
                                      "checkpoint",
                                      "walltime",
                                      defaults::checkpoint::walltime);
    if (walltime_str.empty()) {
      walltime_str = defaults::checkpoint::walltime;
    }
    set("checkpoint.walltime", walltime_str);

    const auto checkpoint_write_path = toml::find_or(
      toml_data,
      "checkpoint",
      "write_path",
      fmt::format(defaults::checkpoint::write_path.c_str(),
                  get<std::string>("simulation.name").c_str()));
    set("checkpoint.write_path", checkpoint_write_path);
    set("checkpoint.read_path",
        toml::find_or(toml_data, "checkpoint", "read_path", checkpoint_write_path));

    /* [adios2] ------------------------------------------------------------- */
    set("adios2.aggregators_per_node",
        toml::find_or<int>(toml_data,
                           "adios2",
                           "aggregators_per_node",
                           defaults::adios2::aggregators_per_node));
    set("adios2.max_shm_size",
        toml::find_or<size_t>(toml_data,
                              "adios2",
                              "max_shm_size",
                              defaults::adios2::max_shm_size));
    set("adios2.buffer_chunk_size",
        toml::find_or<size_t>(toml_data,
                              "adios2",
                              "buffer_chunk_size",
                              defaults::adios2::buffer_chunk_size));

    /* [diagnostics] -------------------------------------------------------- */
    set("diagnostics.interval",
        toml::find_or(toml_data, "diagnostics", "interval", defaults::diag::interval));
    set("diagnostics.blocking_timers",
        toml::find_or(toml_data, "diagnostics", "blocking_timers", false));
    set("diagnostics.colored_stdout",
        toml::find_or(toml_data, "diagnostics", "colored_stdout", false));
    set("diagnostics.log_level",
        toml::find_or(toml_data, "diagnostics", "log_level", defaults::diag::log_level));

    /* inferred variables --------------------------------------------------- */
    params::Boundaries boundaries_params {
      isPromised("grid.boundaries.match.ds"),
      isPromised("grid.boundaries.absorb.ds"),
      isPromised("grid.boundaries.atmosphere.temperature")
    };
    boundaries_params.read(dim, coord_enum, extent_pairwise, toml_data);
    boundaries_params.setParams(this);

    /* [algorithms] --------------------------------------------------------- */
    params::Algorithms                alg_params {};
    const std::map<std::string, bool> alg_extra_flags = {
      {  "gr",          engine_enum == SimEngine::GRPIC },
      { "gca", isPromised("algorithms.gca.e_ovr_b_max") },
    };
    alg_params.read(get<real_t>("scales.dx0"), alg_extra_flags, toml_data);
    alg_params.setParams(alg_extra_flags, this);

    /* hybrid timestep — correct CFL --------------------------------------- */
    // The default light-crossing dt = CFL * dx0 assumes the fastest signal
    // travels at ~1 code-velocity unit (as for SR/GR, where speeds are bounded
    // by c = 1). A hybrid plasma has no displacement current; its fastest
    // signals are the DISPERSIVE whistler/Hall mode and the bulk ion flow, both
    // larger than 1 in code units. Using dt = CFL * dx0 then advects particles
    // by > 1 cell/step (Courant > 1), which breaks the single-cell rebucket and
    // the moment deposit. Replace it with the Pegasus CFL (Kunz, Stone & Bai
    // 2014, §3.5): dt <= CFL * min( dx0 / v_max, 2*pi/Omega ), with the
    // ground-frame max signal speed
    //   v_max = v_flow + v_A + v_whistler,
    //   v_A        = d0 / rho0                      (Alfven speed, code units),
    //   v_whistler = (d0^2 / rho0) * pi / dx0       (grid-scale Hall/whistler;
    //                d0^2/rho0 is the EMF Hall coefficient coeff_2, k_max~pi/dx0),
    // and v_flow an optional user-set characteristic flow speed in code units
    // (hybrid.v_max, default 0 — set it for super-whistler flows).
    if (engine_enum == SimEngine::HYBRID) {
      const auto cfl        = get<real_t>("algorithms.timestep.CFL");
      const auto dx0        = get<real_t>("scales.dx0");
      const auto d0         = get<real_t>("scales.skindepth0");
      const auto rho0       = get<real_t>("scales.larmor0");
      const auto v_flow     = get<real_t>("hybrid.v_max");
      const auto v_alfven   = d0 / rho0;
      const auto v_whistler = (SQR(d0) / rho0) *
                              static_cast<real_t>(constant::PI) / dx0;
      const auto v_max  = v_flow + v_alfven + v_whistler;
      const auto dt_adv = dx0 / v_max;
      // gyration limit 2*pi/Omega (Omega = 1/rho0); rarely binding
      const auto dt_gyr = static_cast<real_t>(constant::TWO_PI) * rho0;
      const auto dt_hyb = cfl * ((dt_adv < dt_gyr) ? dt_adv : dt_gyr);
      set("algorithms.timestep.dt", dt_hyb);
    }

    /* extra physics ------------------------------------------------------ */
    params::Extra                     extra_params {};
    const std::map<std::string, bool> extra_extra_flags = {
      {     "synchrotron_drag",isPromised("radiation.drag.synchrotron.gamma_rad")                               },
      {         "compton_drag",          isPromised("radiation.drag.compton.gamma_rad") },
      { "synchrotron_emission",
       isPromised("radiation.emission.synchrotron.photon_species")                     },
      {     "compton_emission", isPromised("radiation.emission.compton.photon_species") }
    };
    extra_params.read(extra_extra_flags, toml_data, this);
    extra_params.setParams(extra_extra_flags, this);

    // @TODO: disabling stats for non-Cartesian
    if (coord_enum != Coord::Cartesian) {
      set("output.stats.enable", false);
    }
  }

  void SimulationParams::setSetupParams(const toml::value& toml_data) {
    /* [setup] -------------------------------------------------------------- */
    const auto setup = toml::find_or(toml_data, "setup", toml::table {});
    for (const auto& [key, val] : setup) {
      if (val.is_boolean()) {
        set("setup." + key, (bool)(val.as_boolean()));
      } else if (val.is_integer()) {
        set("setup." + key, (int)(val.as_integer()));
      } else if (val.is_floating()) {
        set("setup." + key, (real_t)(val.as_floating()));
      } else if (val.is_string()) {
        set("setup." + key, (std::string)(val.as_string()));
      } else if (val.is_array()) {
        const auto val_arr = val.as_array();
        if (val_arr.empty()) {
          continue;
        } else {
          if (val_arr[0].is_integer()) {
            std::vector<int> vec;
            for (const auto& v : val_arr) {
              vec.push_back(static_cast<int>(v.as_integer()));
            }
            set("setup." + key, vec);
          } else if (val_arr[0].is_floating()) {
            std::vector<real_t> vec;
            for (const auto& v : val_arr) {
              vec.push_back(static_cast<real_t>(v.as_floating()));
            }
            set("setup." + key, vec);
          } else if (val_arr[0].is_boolean()) {
            std::vector<bool> vec;
            for (const auto& v : val_arr) {
              vec.push_back(v.as_boolean());
            }
            set("setup." + key, vec);
          } else if (val_arr[0].is_string()) {
            std::vector<std::string> vec;
            for (const auto& v : val_arr) {
              vec.push_back(v.as_string());
            }
            set("setup." + key, vec);
          } else if (val_arr[0].is_array()) {
            if (val_arr[0].as_array().empty()) {
              raise::Error("empty inner arrays not allowed in [setup]", HERE);
            } else if (val_arr[0][0].is_integer()) {
              std::vector<std::vector<int>> vec;
              for (const auto& v1 : val_arr) {
                std::vector<int> inner_vec;
                for (const auto& v2 : v1.as_array()) {
                  inner_vec.push_back(static_cast<int>(v2.as_integer()));
                }
                vec.push_back(inner_vec);
              }
              set("setup." + key, vec);
            } else if (val_arr[0][0].is_floating()) {
              std::vector<std::vector<real_t>> vec;
              for (const auto& v1 : val_arr) {
                std::vector<real_t> inner_vec;
                for (const auto& v2 : v1.as_array()) {
                  inner_vec.push_back(static_cast<real_t>(v2.as_floating()));
                }
                vec.push_back(inner_vec);
              }
              set("setup." + key, vec);
            } else if (val_arr[0][0].is_boolean()) {
              std::vector<std::vector<bool>> vec;
              for (const auto& v : val_arr) {
                std::vector<bool> inner_vec;
                for (const auto& v2 : v.as_array()) {
                  inner_vec.push_back(v2.as_boolean());
                }
                vec.push_back(inner_vec);
              }
              set("setup." + key, vec);
            } else if (val_arr[0][0].is_string()) {
              std::vector<std::vector<std::string>> vec;
              for (const auto& v : val_arr) {
                std::vector<std::string> inner_vec;
                for (const auto& v2 : v.as_array()) {
                  inner_vec.push_back(v2.as_string());
                }
                vec.push_back(inner_vec);
              }
              set("setup." + key, vec);
            } else if (val_arr[0][0].is_array()) {
              raise::Error("up to 2D arrays allowed in [setup]", HERE);
            }
          } else {
            raise::Error("invalid setup variable type", HERE);
          }
        }
      }
    }
  }

  void SimulationParams::setCheckpointParams(bool       is_resuming,
                                             timestep_t start_step,
                                             simtime_t  start_time) {
    set("checkpoint.is_resuming", is_resuming);
    set("checkpoint.start_step", start_step);
    set("checkpoint.start_time", start_time);
  }

  void SimulationParams::checkPromises() const {
    raise::ErrorIf(!promisesFulfilled(),
                   "Have not defined all the necessary variables",
                   HERE);
  }

  void SimulationParams::saveTOML(const std::string& path, simtime_t time) const {
    CallOnce([&]() {
      std::ofstream metadata;
      metadata.open(path);
      metadata << fmt::format("[metadata]\n  time = %f\n\n", time) << data()
               << '\n';
      metadata.close();
    });
  }

} // namespace ntt
