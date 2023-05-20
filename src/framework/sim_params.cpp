#include "sim_params.h"

#include "wrapper.h"

#include "input.h"
#include "output.h"
#include "particles.h"
#include "qmath.h"
#include "utils.h"

#include <toml.hpp>

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace ntt {

  SimulationParams::SimulationParams(const toml::value& inputdata, Dimension dim) {
    m_inputdata          = inputdata;
    m_title              = get<std::string>("simulation", "title", defaults::title);
    m_total_runtime      = get<real_t>("simulation", "runtime", defaults::runtime);
    m_correction         = get<real_t>("algorithm", "correction", defaults::correction);
    m_enable_fieldsolver = get<bool>("algorithm", "fieldsolver_ON", true);
    m_enable_deposit     = get<bool>("algorithm", "deposit_ON", true);

    // reading particle parameters
    auto nspec           = get<int>("particles", "n_species", defaults::n_species);
    for (int i { 0 }; i < nspec; ++i) {
      auto label = get<std::string>(
        "species_" + std::to_string(i + 1), "label", "s" + std::to_string(i + 1));
      auto mass   = get<float>("species_" + std::to_string(i + 1), "mass");
      auto charge = get<float>("species_" + std::to_string(i + 1), "charge");
      auto maxnpart
        = (std::size_t)(get<double>("species_" + std::to_string(i + 1), "maxnpart"));
      auto pusher_str = get<std::string>(
        "species_" + std::to_string(i + 1),
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
    m_shuffle_interval = get<int>("particles", "shuffle_step", defaults::shuffle_interval);
    m_max_dead_frac    = get<double>("particles", "max_dead_frac", defaults::max_dead_frac);
    m_use_weights      = get<bool>("particles", "use_weights", defaults::use_weights);

#ifdef MINKOWSKI_METRIC
    m_metric = "minkowski";
#elif defined(SPHERICAL_METRIC)
    m_metric = "spherical";
#elif defined(QSPHERICAL_METRIC)
    m_metric = "qspherical";
#elif defined(KERR_SCHILD_METRIC)
    m_metric = "kerr_schild";
#elif defined(QKERR_SCHILD_METRIC)
    m_metric = "qkerr_schild";
#else
    NTTHostError("unrecognized metric");
#endif

    // domain size / resolution
    m_resolution = get<std::vector<unsigned int>>("domain", "resolution");
    m_extent     = get<std::vector<real_t>>("domain", "extent");
    if (m_metric == "minkowski") {
      // minkowski
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
    } else if ((m_metric == "spherical") || (m_metric == "qspherical")
               || (m_metric == "kerr_schild") || (m_metric == "qkerr_schild")) {
      // spherical (quasi-spherical) grid
      NTTHostErrorIf((m_extent.size() < 2), "not enough values in `extent` input");
      m_extent.erase(m_extent.begin() + 2, m_extent.end());
      if ((m_metric == "qspherical") || (m_metric == "qkerr_schild")) {
        m_metric_parameters[0] = get<real_t>("domain", "qsph_r0");
        m_metric_parameters[1] = get<real_t>("domain", "qsph_h");
        NTTHostErrorIf((AlmostEqual(m_metric_parameters[1], ZERO)), "qsph_h must be non-zero");
      }
      m_metric_parameters[2] = get<real_t>("domain", "sph_rabsorb");
      m_metric_parameters[3] = get<real_t>("domain", "absorb_coeff", (real_t)(1.0));

      if ((m_metric == "kerr_schild") || (m_metric == "qkerr_schild")) {
        real_t spin { get<real_t>("domain", "a") };
        real_t rh { ONE + math::sqrt(ONE - spin * spin) };
        m_metric_parameters[4] = spin;
        // m_extent[0] *= rh;
        // m_extent[1] *= rh;
        // m_metric_parameters[2] *= rh;
        m_metric_parameters[5] = rh;
      }

      m_extent.push_back(0.0);
      m_extent.push_back(constant::PI);
      m_extent.push_back(0.0);
      m_extent.push_back(constant::TWO_PI);
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

    m_cfl        = get<real_t>("algorithm", "CFL", defaults::cfl);
    assert(m_cfl > 0);

    // number of current filter passes
    m_current_filters
      = get<unsigned short>("algorithm", "current_filters", defaults::current_filters);

    // output params
    m_output_format
      = get<std::string>("output", "format", defaults::output_format, options::outputs);
    m_output_interval   = get<int>("output", "interval", defaults::output_interval);
    m_output_fields     = get<std::vector<std::string>>("output", "fields");
    m_output_particles  = get<std::vector<std::string>>("output", "particles");
    m_output_mom_smooth = get<int>("output", "mom_smooth", defaults::output_mom_smooth);
    m_output_prtl_stride
      = get<std::size_t>("output", "prtl_stride", defaults::output_prtl_stride);

    // diagnostic params
    m_diag_interval   = get<int>("diagnostics", "interval", defaults::diag_interval);
    m_blocking_timers = get<bool>("diagnostics", "blocking_timers", defaults::blocking_timers);
  }
}    // namespace ntt