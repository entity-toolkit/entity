#include "global.h"
#include "sim_params.h"
#include "input.h"
#include "particles.h"

#include <toml/toml.hpp>

#include <stdexcept>
#include <string>
#include <cassert>
#include <vector>

namespace ntt {

  SimulationParams::SimulationParams(const toml::value& inputdata, Dimension dim) {
    m_inputdata = inputdata;
    m_title = readFromInput<std::string>(m_inputdata, "simulation", "title", defaults::title);
    m_total_runtime = readFromInput<real_t>(m_inputdata, "simulation", "runtime");
    m_correction = readFromInput<real_t>(m_inputdata, "algorithm", "correction");

    // reading particle parameters
    auto nspec = readFromInput<int>(m_inputdata, "particles", "n_species", defaults::n_species);
    for (int i {0}; i < nspec; ++i) {
      auto label = readFromInput<std::string>(
        m_inputdata, "species_" + std::to_string(i + 1), "label", "s" + std::to_string(i + 1));
      auto mass = readFromInput<float>(m_inputdata, "species_" + std::to_string(i + 1), "mass");
      auto charge = readFromInput<float>(m_inputdata, "species_" + std::to_string(i + 1), "charge");
      auto maxnpart = (std::size_t)(readFromInput<double>(m_inputdata, "species_" + std::to_string(i + 1), "maxnpart"));
      auto pusher_str
        = readFromInput<std::string>(m_inputdata, "species_" + std::to_string(i + 1), "pusher", defaults::pusher);
      ParticlePusher pusher {ParticlePusher::UNDEFINED};
      if ((mass == 0.0) && (charge == 0.0)) {
        pusher = ParticlePusher::PHOTON;
      } else if (pusher_str == "Vay") {
        pusher = ParticlePusher::VAY;
      } else if (pusher_str == "Boris") {
        pusher = ParticlePusher::BORIS;
      }
      m_species.emplace_back(ParticleSpecies(label, mass, charge, maxnpart, pusher));
    }

#if METRIC == MINKOWSKI_METRIC
    m_metric = "minkowski";
#elif METRIC == SPHERICAL_METRIC
    m_metric = "spherical";
#elif METRIC == QSPHERICAL_METRIC
    m_metric = "qspherical";
#elif METRIC == KERR_SCHILD_METRIC
    m_metric = "kerr_schild";
#else
    NTTError("unrecognized metric");
#endif

    // domain size / resolution
    m_resolution = readFromInput<std::vector<std::size_t>>(m_inputdata, "domain", "resolution");
    m_extent = readFromInput<std::vector<real_t>>(m_inputdata, "domain", "extent");
    if (m_metric == "minkowski") {
      // minkowski
      if (((short)(m_resolution.size()) < (short)(dim)) || ((short)(m_extent.size()) < 2 * (short)(dim))) {
        NTTError("not enough values in `extent` or `resolution` input");
      }
      // enforce dx = dy = dz
      auto dx {(m_extent[1] - m_extent[0]) / (real_t)(m_resolution[0])};
      if (m_resolution.size() > 1) {
        auto dy {(m_extent[3] - m_extent[2]) / (real_t)(m_resolution[1])};
        if (dx != dy) { NTTError("dx != dy in minkowski"); }
      }
      if (m_resolution.size() > 2) {
        auto dz {(m_extent[5] - m_extent[4]) / (real_t)(m_resolution[2])};
        if (dx != dz) { NTTError("dx != dz in minkowski"); }
      }
    } else if ((m_metric == "spherical") || (m_metric == "qspherical")) {
      // spherical (quasi-spherical) grid
      if (m_extent.size() < 2) {
        NTTError("not enough values in `extent` input"); 
      }
      m_extent.erase(m_extent.begin() + 2, m_extent.end());
      if (m_metric == "qspherical") {
        m_metric_parameters[0] = readFromInput<real_t>(inputdata, "domain", "qsph_r0");
        m_metric_parameters[1] = readFromInput<real_t>(inputdata, "domain", "qsph_h");
      }
      m_metric_parameters[2] = readFromInput<real_t>(inputdata, "domain", "sph_rabsorb");

      if (m_metric == "kerr_schild") {
        m_metric_parameters[3] = readFromInput<real_t>(inputdata, "domain", "a");
      }

      m_extent.push_back(0.0);
      m_extent.push_back(constant::PI);
      m_extent.push_back(0.0);
      m_extent.push_back(constant::TWO_PI);
    }
    // leave only necessary extent/resolution (<= DIM)
    m_extent.erase(m_extent.begin() + 2 * (short)(dim), m_extent.end());
    m_resolution.erase(m_resolution.begin() + (short)(dim), m_resolution.end());

    if (m_metric == "minkowski") {
      auto boundaries = readFromInput<std::vector<std::string>>(m_inputdata, "domain", "boundaries");
      short b {0};
      for (auto& bc : boundaries) {
        if (bc == "PERIODIC") {
          m_boundaries.push_back(BoundaryCondition::PERIODIC);
        } else if (bc == "OPEN") {
          m_boundaries.push_back(BoundaryCondition::OPEN);
        } else if (bc == "USER") {
          m_boundaries.push_back(BoundaryCondition::USER);
        } else {
          m_boundaries.push_back(BoundaryCondition::UNDEFINED);
        }
        ++b;
        if (b >= (short)(dim)) { break; }
      }
    } else if ((m_metric == "spherical") || (m_metric == "qspherical")) {
      // rmin, rmax boundaries only
      m_boundaries.push_back(BoundaryCondition::USER);
      m_boundaries.push_back(BoundaryCondition::USER);
    } else {
      NTTError("coordinate system not implemented");
    }

    // plasma params
    m_ppc0 = readFromInput<real_t>(m_inputdata, "units", "ppc0");
    m_larmor0 = readFromInput<real_t>(m_inputdata, "units", "larmor0");
    m_skindepth0 = readFromInput<real_t>(m_inputdata, "units", "skindepth0");
    m_sigma0 = m_larmor0 * m_larmor0 / (m_skindepth0 * m_skindepth0);
    m_charge0 = 1.0 / (m_ppc0 * m_skindepth0 * m_skindepth0);
    m_B0 = 1.0 / m_larmor0;

    m_cfl = readFromInput<real_t>(m_inputdata, "algorithm", "CFL", defaults::cfl);
    assert(m_cfl > 0);
  }
} // namespace ntt
