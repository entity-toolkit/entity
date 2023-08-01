#include "sim_params.h"

#include "wrapper.h"

#include "io/input.h"
#include "io/output.h"
#include "meshblock/particles.h"
#include "utils/qmath.h"
#include "utils/utils.h"

#include <toml.hpp>

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace ntt {

  SimulationParams::SimulationParams(const toml::value& inputdata, Dimension dim) {
    m_inputdata          = inputdata;
    m_title              = get("simulation", "title", defaults::title);
    m_total_runtime      = get("simulation", "runtime", defaults::runtime);
    m_correction         = get("algorithm", "correction", defaults::correction);
    m_enable_fieldsolver = get("algorithm", "fieldsolver_ON", true);
    m_enable_deposit     = get("algorithm", "deposit_ON", true);

    // reading particle parameters
    auto nspec           = get("particles", "n_species", defaults::n_species);
    for (int i { 0 }; i < nspec; ++i) {
      auto label
        = get("species_" + std::to_string(i + 1), "label", "s" + std::to_string(i + 1));
      auto mass   = get<float>("species_" + std::to_string(i + 1), "mass");
      auto charge = get<float>("species_" + std::to_string(i + 1), "charge");
      auto maxnpart
        = (std::size_t)(get<double>("species_" + std::to_string(i + 1), "maxnpart"));
      auto pusher_str
        = get("species_" + std::to_string(i + 1),
              "pusher",
              (mass == 0.0) && (charge == 0.0) ? defaults::ph_pusher : defaults::em_pusher,
              options::pushers);
      ParticlePusher pusher { ParticlePusher::UNDEFINED };
      if (pusher_str == "Photon") {
        pusher = ParticlePusher::PHOTON;
      } else if (pusher_str == "Vay") {
        pusher = ParticlePusher::VAY;
      } else if (pusher_str == "Boris") {
        pusher = ParticlePusher::BORIS;
      } else if (pusher_str == "None") {
        pusher = ParticlePusher::NONE;
      } else {
        NTTHostError("unrecognized pusher");
      }
      m_species.emplace_back(ParticleSpecies(i + 1, label, mass, charge, maxnpart, pusher));
    }
    m_use_weights      = get("particles", "use_weights", defaults::use_weights);

    m_metric           = SIMULATION_METRIC;
    if (m_metric == "minkowski") {
      m_coordinates = "cartesian";
    } else if (m_metric[0] == 'q') {
      m_coordinates = "qspherical";
    } else {
      m_coordinates = "spherical";
    }

    // domain decomposition
    m_domaindecomposition
      = get<std::vector<unsigned int>>("domain", "decomposition", std::vector<unsigned int>());

    // domain size / resolution
    m_resolution = get<std::vector<unsigned int>>("domain", "resolution");
    m_extent     = get<std::vector<real_t>>("domain", "extent");
    if (m_coordinates == "cartesian") {
      /**
       *  cartesian grid
       */
      NTTHostErrorIf((((short)(m_resolution.size()) < (short)(dim))
                      || ((short)(m_extent.size()) < 2 * (short)(dim))),
                     "not enough values in `extent` or `resolution` input");
      // enforce dx = dy = dz
      auto dx { (m_extent[1] - m_extent[0]) / (real_t)(m_resolution[0]) };
      if (m_resolution.size() > 1) {
        auto dy { (m_extent[3] - m_extent[2]) / (real_t)(m_resolution[1]) };
        NTTHostErrorIf((dx != dy), "dx != dy in minkowski");
      }
      if (m_resolution.size() > 2) {
        auto dz { (m_extent[5] - m_extent[4]) / (real_t)(m_resolution[2]) };
        NTTHostErrorIf((dx != dz), "dx != dz in minkowski");
      }
    } else if (m_coordinates.find("spherical") != std::string::npos) {
      /**
       * spherical (quasi-spherical) grid
       */
      NTTHostErrorIf((m_extent.size() < 2), "not enough values in `extent` input");
      m_extent.erase(m_extent.begin() + 2, m_extent.end());
      if (m_coordinates == "qspherical") {
        m_metric_parameters[0] = get<real_t>("domain", "qsph_r0");
        m_metric_parameters[1] = get<real_t>("domain", "qsph_h");
      }
      m_metric_parameters[2] = get("domain", "sph_rabsorb", (real_t)(m_extent[1] * 0.9));
      m_metric_parameters[3] = get("domain", "absorb_coeff", (real_t)(1.0));

      // GR specific
      if (m_metric.find("kerr_schild") != std::string::npos) {
        real_t spin { get("domain", "a", ZERO) };
        real_t rh { ONE + math::sqrt(ONE - spin * spin) };
        m_metric_parameters[4] = spin;
        m_metric_parameters[5] = rh;

        m_gr_pusher_epsilon    = get("algorithm", "gr_pusher_epsilon", (real_t)(1.0e-6));
        m_gr_pusher_niter      = get("algorithm", "gr_pusher_niter", 10);
      }

      m_extent.push_back(ZERO);
      m_extent.push_back(constant::PI);
      m_extent.push_back(ZERO);
      m_extent.push_back(constant::TWO_PI);
    } else {
      NTTHostError("unrecognized coordinates: " + m_coordinates);
    }
    // leave only necessary extent/resolution (<= DIM)
    m_extent.erase(m_extent.begin() + 2 * (short)(dim), m_extent.end());
    m_resolution.erase(m_resolution.begin() + (short)(dim), m_resolution.end());

    auto  boundaries = get<std::vector<std::vector<std::string>>>("domain", "boundaries");
    short b { 0 };
    for (auto& bc_xi : boundaries) {
      std::vector<BoundaryCondition> boundaries_xi;
      for (auto& bc : bc_xi) {
        TestValidOption(bc, options::boundaries);
        if (bc == "PERIODIC") {
          boundaries_xi.push_back(BoundaryCondition::PERIODIC);
        } else if (bc == "ABSORB") {
          boundaries_xi.push_back(BoundaryCondition::ABSORB);
        } else if (bc == "OPEN") {
          boundaries_xi.push_back(BoundaryCondition::OPEN);
        } else if (bc == "CUSTOM") {
          boundaries_xi.push_back(BoundaryCondition::CUSTOM);
        } else if (bc == "AXIS") {
          boundaries_xi.push_back(BoundaryCondition::AXIS);
        } else {
          boundaries_xi.push_back(BoundaryCondition::UNDEFINED);
        }
      }
      m_boundaries.push_back(boundaries_xi);
      ++b;
      if (b >= (short)(dim)) {
        break;
      }
    }

    // plasma params
    m_ppc0       = get<real_t>("units", "ppc0");
    m_larmor0    = get<real_t>("units", "larmor0");
    m_skindepth0 = get<real_t>("units", "skindepth0");
    m_sigma0     = SQR(m_skindepth0) / SQR(m_larmor0);

    // if dt not specified (== -1), will use CFL to calculate it
    m_dt         = get("algorithm", "dt", -ONE);
    m_cfl        = get("algorithm", "CFL", defaults::cfl);
    assert(m_cfl > 0);

    // number of current filter passes
    m_current_filters = get("algorithm", "current_filters", defaults::current_filters);

    // output params
    m_output_format   = get("output", "format", defaults::output_format, options::outputs);
    m_output_interval = get("output", "interval", defaults::output_interval);
    m_output_interval_time = get("output", "interval_time", -1.0);
    m_output_fields
      = get<std::vector<std::string>>("output", "fields", std::vector<std::string>());
    m_output_particles
      = get<std::vector<std::string>>("output", "particles", std::vector<std::string>());
    m_output_mom_smooth  = get("output", "mom_smooth", defaults::output_mom_smooth);
    m_output_prtl_stride = get("output", "prtl_stride", defaults::output_prtl_stride);
    m_output_as_is       = get("output", "as_is", false);
    m_output_ghosts      = get("output", "ghosts", false);

    // diagnostic params
    m_diag_interval      = get("diagnostics", "interval", defaults::diag_interval);
    m_blocking_timers    = get("diagnostics", "blocking_timers", defaults::blocking_timers);
  }
}    // namespace ntt