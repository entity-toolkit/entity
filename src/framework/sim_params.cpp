#include "sim_params.h"

#include "wrapper.h"

#include "input.h"
#include "particles.h"
#include "qmath.h"

#include <toml/toml.hpp>

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace ntt {

  SimulationParams::SimulationParams(const toml::value& inputdata, Dimension dim) {
    m_inputdata = inputdata;
    m_title = readFromInput<std::string>(m_inputdata, "simulation", "title", defaults::title);
    m_total_runtime
      = readFromInput<real_t>(m_inputdata, "simulation", "runtime", defaults::runtime);
    m_correction
      = readFromInput<real_t>(m_inputdata, "algorithm", "correction", defaults::correction);
    m_enable_fieldsolver
      = readFromInput<bool>(m_inputdata, "algorithm", "fieldsolver_ON", true);
    m_enable_deposit = readFromInput<bool>(m_inputdata, "algorithm", "deposit_ON", true);

    // reading particle parameters
    auto nspec
      = readFromInput<int>(m_inputdata, "particles", "n_species", defaults::n_species);
    for (int i { 0 }; i < nspec; ++i) {
      auto label = readFromInput<std::string>(
        m_inputdata, "species_" + std::to_string(i + 1), "label", "s" + std::to_string(i + 1));
      auto mass
        = readFromInput<float>(m_inputdata, "species_" + std::to_string(i + 1), "mass");
      auto charge
        = readFromInput<float>(m_inputdata, "species_" + std::to_string(i + 1), "charge");
      auto maxnpart = (std::size_t)(
        readFromInput<double>(m_inputdata, "species_" + std::to_string(i + 1), "maxnpart"));
      auto pusher_str = readFromInput<std::string>(
        m_inputdata, "species_" + std::to_string(i + 1), "pusher", defaults::pusher);
      ParticlePusher pusher { ParticlePusher::UNDEFINED };
      if ((mass == 0.0) && (charge == 0.0)) {
        pusher = ParticlePusher::PHOTON;
      } else if (pusher_str == "Vay") {
        pusher = ParticlePusher::VAY;
      } else if (pusher_str == "Boris") {
        pusher = ParticlePusher::BORIS;
      }
      m_species.emplace_back(ParticleSpecies(i + 1, label, mass, charge, maxnpart, pusher));
    }
    m_shuffle_interval = readFromInput<int>(
      m_inputdata, "particles", "shuffle_step", defaults::shuffle_interval);
    m_max_dead_frac = readFromInput<float>(
      m_inputdata, "particles", "max_dead_frac", defaults::max_dead_frac);

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
    m_resolution
      = readFromInput<std::vector<unsigned int>>(m_inputdata, "domain", "resolution");
    m_extent = readFromInput<std::vector<real_t>>(m_inputdata, "domain", "extent");
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
        m_metric_parameters[0] = readFromInput<real_t>(inputdata, "domain", "qsph_r0");
        m_metric_parameters[1] = readFromInput<real_t>(inputdata, "domain", "qsph_h");
        NTTHostErrorIf((AlmostEqual(m_metric_parameters[1], ZERO)), "qsph_h must be non-zero");
      }
      m_metric_parameters[2] = readFromInput<real_t>(inputdata, "domain", "sph_rabsorb");
      m_metric_parameters[3]
        = readFromInput<real_t>(inputdata, "domain", "absorb_coeff", (real_t)(1.0));

      if ((m_metric == "kerr_schild") || (m_metric == "qkerr_schild")) {
        real_t spin { readFromInput<real_t>(inputdata, "domain", "a") };
        real_t rh { ONE + math::sqrt(ONE - spin * spin) };
        m_metric_parameters[4] = spin;
        m_extent[0] *= rh;
        m_extent[1] *= rh;
        m_metric_parameters[2] *= rh;
      }

      m_extent.push_back(0.0);
      m_extent.push_back(constant::PI);
      m_extent.push_back(0.0);
      m_extent.push_back(constant::TWO_PI);
    }
    // leave only necessary extent/resolution (<= DIM)
    m_extent.erase(m_extent.begin() + 2 * (short)(dim), m_extent.end());
    m_resolution.erase(m_resolution.begin() + (short)(dim), m_resolution.end());

    auto boundaries
      = readFromInput<std::vector<std::string>>(m_inputdata, "domain", "boundaries");
    short b { 0 };
    for (auto& bc : boundaries) {
      if (bc == "PERIODIC") {
        m_boundaries.push_back(BoundaryCondition::PERIODIC);
      } else if (bc == "ABSORB") {
        m_boundaries.push_back(BoundaryCondition::ABSORB);
      } else if (bc == "OPEN") {
        m_boundaries.push_back(BoundaryCondition::OPEN);
      } else if (bc == "USER") {
        m_boundaries.push_back(BoundaryCondition::USER);
      } else {
        m_boundaries.push_back(BoundaryCondition::UNDEFINED);
      }
      ++b;
      if (b >= (short)(dim)) {
        break;
      }
    }

    // plasma params
    m_ppc0       = readFromInput<real_t>(m_inputdata, "units", "ppc0");
    m_larmor0    = readFromInput<real_t>(m_inputdata, "units", "larmor0");
    m_skindepth0 = readFromInput<real_t>(m_inputdata, "units", "skindepth0");
    m_sigma0     = SQR(m_skindepth0) / SQR(m_larmor0);

    m_cfl        = readFromInput<real_t>(m_inputdata, "algorithm", "CFL", defaults::cfl);
    assert(m_cfl > 0);

    // number of current filter passes
    m_current_filters = readFromInput<unsigned short>(
      m_inputdata, "algorithm", "current_filters", defaults::current_filters);

    // output params
    m_output_format
      = readFromInput<std::string>(m_inputdata, "output", "format", defaults::output_format);
    m_output_interval
      = readFromInput<int>(m_inputdata, "output", "interval", defaults::output_interval);
  }

  template <typename T>
  auto SimulationParams::get<T>(const std::string& block,
                                const std::string& key,
                                const T&           defval) const -> T {
    return readFromInput<T>(m_inputdata, block, key, defval);
  }

  template <typename T>
  auto SimulationParams::get<T>(const std::string& block, const std::string& key) const -> T {
    return readFromInput<T>(m_inputdata, block, key);
  }

}    // namespace ntt

template auto ntt::SimulationParams::get<float>(const std::string&,
                                                const std::string&,
                                                const float&) const -> float;
template auto ntt::SimulationParams::get<float>(const std::string&, const std::string&) const
  -> float;
template auto ntt::SimulationParams::get<double>(const std::string&,
                                                 const std::string&,
                                                 const double&) const -> double;
template auto ntt::SimulationParams::get<double>(const std::string&, const std::string&) const
  -> double;
template auto ntt::SimulationParams::get<int>(const std::string&,
                                              const std::string&,
                                              const int&) const -> int;
template auto ntt::SimulationParams::get<int>(const std::string&, const std::string&) const
  -> int;
template auto ntt::SimulationParams::get<bool>(const std::string&,
                                               const std::string&,
                                               const bool&) const -> bool;
template auto ntt::SimulationParams::get<bool>(const std::string&, const std::string&) const
  -> bool;
