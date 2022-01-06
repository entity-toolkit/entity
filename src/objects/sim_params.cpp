#include "global.h"
#include "sim_params.h"
#include "cargs.h"
#include "input.h"
#include "particles.h"

#include <toml/toml.hpp>

#include <stdexcept>
#include <string>
#include <cassert>
#include <vector>
#include <iostream>

namespace ntt {

  SimulationParams::SimulationParams(const toml::value& inputdata, Dimension dim) {
    m_inputdata = inputdata;

    m_title = readFromInput<std::string>(m_inputdata, "simulation", "title", "PIC_Sim");
    m_runtime = readFromInput<real_t>(m_inputdata, "simulation", "runtime");
    m_correction = readFromInput<real_t>(m_inputdata, "algorithm", "correction");

    // particle parameters
    auto nspec = readFromInput<int>(m_inputdata, "particles", "n_species", 0);
    for (int i {0}; i < nspec; ++i) {
      auto label = readFromInput<std::string>(m_inputdata,
                                              "species_" + std::to_string(i + 1),
                                              "label",
                                              "s" + std::to_string(i + 1));
      auto mass = readFromInput<float>(m_inputdata,
                                       "species_" + std::to_string(i + 1),
                                       "mass");
      auto charge = readFromInput<float>(m_inputdata,
                                         "species_" + std::to_string(i + 1),
                                         "charge");
      auto maxnpart = (std::size_t)(readFromInput<double>(m_inputdata,
                                                          "species_" + std::to_string(i + 1),
                                                          "maxnpart"));
      auto pusher_str = readFromInput<std::string>(m_inputdata,
                                                   "species_" + std::to_string(i + 1),
                                                   "pusher",
                                                   "Boris");
      ParticlePusher pusher {UNDEFINED_PUSHER};
      if ((mass == 0.0) && (charge == 0.0)) {
        pusher = PHOTON_PUSHER;
      } else if (pusher_str == "Vay") {
        pusher = VAY_PUSHER;
      } else if (pusher_str == "Boris") {
        pusher = BORIS_PUSHER;
      }
      m_species.emplace_back(ParticleSpecies(label, mass, charge, maxnpart, pusher));
    }
    m_prtl_shape = (ParticleShape)(readFromInput<short>(m_inputdata,
                                                        "algorithm",
                                                        "particle_shape",
                                                        FIRST_ORDER));

    // hardcoded PIC regime
    m_simtype = PIC_SIM;

#ifndef CURVILINEAR_COORDS
    m_coord_system = "cartesian";
    if (readFromInput<std::string>(m_inputdata, "domain", "coord_sys", "cartesian") != "cartesian") {
      throw std::invalid_argument("Non-cartesian coordinates specified but the code is not compiled with the `-curv` flag.");
    }
#else
    m_coord_system = readFromInput<std::string>(m_inputdata, "domain", "coord_sys");
#endif

    // box size/resolution
    m_resolution = readFromInput<std::vector<std::size_t>>(m_inputdata, "domain", "resolution");
    m_extent = readFromInput<std::vector<real_t>>(m_inputdata, "domain", "extent");
    if (m_coord_system == "cartesian") {
      // cartesian grid
      if (((short)(m_resolution.size()) < (short)(dim)) ||
          ((short)(m_extent.size()) < 2 * (short)(dim))) {
        throw std::invalid_argument("Not enough values in `extent` or `resolution` input.");
      }
      // enforce dx = dy = dz
      auto dx {(m_extent[1] - m_extent[0]) / (real_t)(m_resolution[0])};
      if (m_resolution.size() > 1) {
        auto dy {(m_extent[3] - m_extent[2]) / (real_t)(m_resolution[1])};
        if (dx != dy) {
          throw std::invalid_argument("dx != dy in cartesian");
        }
      }
      if (m_resolution.size() > 2) {
        auto dz {(m_extent[5] - m_extent[4]) / (real_t)(m_resolution[2])};
        if (dx != dz) {
          throw std::invalid_argument("dx != dz in cartesian");
        }
      }
    } else if ((m_coord_system == "spherical") ||
               (m_coord_system == "qspherical")) {
      // spherical (quasi-spherical) grid
      if (m_extent.size() < 2) {
        throw std::invalid_argument("Not enough values in `extent` input.");
      }
      m_extent.erase(m_extent.begin() + 2, m_extent.end());
      if (m_coord_system == "qspherical") {
        m_coord_parameters[0] = readFromInput<real_t>(inputdata, "domain", "qsph_r0");
        m_coord_parameters[1] = readFromInput<real_t>(inputdata, "domain", "qsph_h");
      }
      m_coord_parameters[2] = readFromInput<real_t>(inputdata, "domain", "sph_rabsorb");
      m_extent.push_back(0.0);
      m_extent.push_back(PI);
      m_extent.push_back(0.0);
      m_extent.push_back(TWO_PI);
    }
    m_extent.erase(m_extent.begin() + 2 * (short)(dim), m_extent.end());
    m_resolution.erase(m_resolution.begin() + (short)(dim), m_resolution.end());

    if (m_coord_system == "cartesian") {
      auto boundaries = readFromInput<std::vector<std::string>>(m_inputdata, "domain", "boundaries");
      short b {0};
      for (auto& bc : boundaries) {
        if (bc == "PERIODIC") {
          m_boundaries.push_back(PERIODIC_BC);
        } else if (bc == "OPEN") {
          m_boundaries.push_back(OPEN_BC);
        } else if (bc == "USER") {
          m_boundaries.push_back(USER_BC);
        } else {
          m_boundaries.push_back(UNDEFINED_BC);
        }
        ++b;
        if (b >= (short)(dim)) { break; }
      }
    } else if ((m_coord_system == "spherical") || (m_coord_system == "qspherical")) {
      // rmin, rmax boundaries only
      m_boundaries.push_back(USER_BC);
      m_boundaries.push_back(USER_BC);
    } else {
      throw std::logic_error("# coordinate system NOT IMPLEMENTED.");
    }

    // plasma params
    m_ppc0 = readFromInput<real_t>(m_inputdata, "units", "ppc0");
    m_larmor0 = readFromInput<real_t>(m_inputdata, "units", "larmor0");
    m_skindepth0 = readFromInput<real_t>(m_inputdata, "units", "skindepth0");
    m_sigma0 = m_larmor0 * m_larmor0 / (m_skindepth0 * m_skindepth0);
    m_charge0 = 1.0 / (m_ppc0 * m_skindepth0 * m_skindepth0);
    m_B0 = 1.0 / m_larmor0;

    m_cfl = readFromInput<real_t>(m_inputdata, "algorithm", "CFL", 0.95);
    assert(m_cfl > 0);
  }

  void SimulationParams::verify() {
    if (m_simtype == UNDEFINED_SIM) {
      throw std::logic_error("# Error: simulation type unspecified.");
    }
    for (auto& b : m_boundaries) {
      if (b == UNDEFINED_BC) { throw std::logic_error("# Error: boundary conditions unspecified."); }
    }
    // TODO: maybe some other tests
  }

} // namespace ntt
