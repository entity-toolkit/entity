#include "global.h"
#include "sim_params.h"
#include "cargs.h"
#include "input.h"
#include "particles.h"

#include <toml/toml.hpp>

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace ntt {
SimulationParams::SimulationParams(const toml::value &inputdata, short dim) {
  m_inputdata = inputdata;

  m_title = readFromInput<std::string>(m_inputdata, "simulation", "title", "PIC_Sim");
  m_runtime = readFromInput<real_t>(m_inputdata, "simulation", "runtime");
  m_timestep = readFromInput<real_t>(m_inputdata, "algorithm", "timestep");
  m_correction = readFromInput<real_t>(m_inputdata, "algorithm", "correction");

  auto nspec = readFromInput<int>(m_inputdata, "particles", "n_species");
  for (int i{0}; i < nspec; ++i) {
    auto label = readFromInput<std::string>(
        m_inputdata, "species_" + std::to_string(i + 1), "label", "s" + std::to_string(i + 1));
    auto mass = readFromInput<float>(m_inputdata, "species_" + std::to_string(i + 1), "mass");
    auto charge = readFromInput<float>(m_inputdata, "species_" + std::to_string(i + 1), "charge");
    auto maxnpart =
        static_cast<std::size_t>(readFromInput<double>(m_inputdata, "species_" + std::to_string(i + 1), "maxnpart"));
    auto pusher_ = readFromInput<std::string>(m_inputdata, "species_" + std::to_string(i + 1), "pusher", "Boris");
    ParticlePusher pusher{UNDEFINED_PUSHER};
    if ((mass == 0.0) && (charge == 0.0)) {
      pusher = PHOTON_PUSHER;
    } else if (pusher_ == "Vay") {
      pusher = VAY_PUSHER;
    } else if (pusher_ == "Boris") {
      pusher = BORIS_PUSHER;
    }
    m_species.emplace_back(ParticleSpecies(label, mass, charge, maxnpart, pusher));
  }

  // TODO: for now only PIC
  m_simtype = PIC_SIM;

  auto coords = readFromInput<std::string>(m_inputdata, "domain", "coord_system", "XYZ");
  if (coords == "X") {
    assert(dim == 1);
    m_coord_system = CARTESIAN_COORD;
  } else if (coords == "XY") {
    assert(dim != 3);
    m_coord_system = CARTESIAN_COORD;
  } else if (coords == "XYZ") {
    m_coord_system = CARTESIAN_COORD;
  } else if (coords == "R_PHI") {
    assert(dim == 2);
    m_coord_system = POLAR_R_PHI_COORD;
  } else if (coords == "R_THETA") {
    assert(dim == 2);
    m_coord_system = POLAR_R_THETA_COORD;
  } else if (coords == "R_THETA_PHI") {
    assert(dim == 3);
    m_coord_system = SPHERICAL_COORD;
  } else if (coords == "logR_THETA_PHI") {
    assert(dim == 3);
    m_coord_system = LOG_SPHERICAL_COORD;
  } else {
    throw std::invalid_argument("Unknown coordinate system specified in the input.");
  }

  // box size/resolution
  m_resolution = readFromInput<std::vector<std::size_t>>(m_inputdata, "domain", "resolution");
  m_extent = readFromInput<std::vector<real_t>>(m_inputdata, "domain", "extent", {0.0, 1.0, 0.0, 1.0, 0.0, 1.0});

  if ((static_cast<short>(m_resolution.size()) < dim) || (static_cast<short>(m_extent.size()) < 2 * dim)) {
    throw std::invalid_argument("Not enough values in `extent` or `resolution` input.");
  }

  m_resolution.erase(m_resolution.begin() + dim, m_resolution.end());
  m_extent.erase(m_extent.begin() + 2 * dim, m_extent.end());

  auto boundaries = readFromInput<std::vector<std::string>>(
      m_inputdata, "domain", "boundaries", {"PERIODIC", "PERIODIC", "PERIODIC"});
  short b{0};
  for (auto &bc : boundaries) {
    if (bc == "PERIODIC") {
      m_boundaries.push_back(PERIODIC_BC);
    } else if (bc == "OPEN") {
      m_boundaries.push_back(OPEN_BC);
    } else {
      m_boundaries.push_back(UNDEFINED_BC);
    }
    ++b;
    if (b >= dim) {
      break;
    }
  }
  // plasma params
  m_ppc0 = readFromInput<real_t>(m_inputdata, "algorithm", "ppc0");
  m_larmor0 = readFromInput<real_t>(m_inputdata, "algorithm", "larmor0");
  m_skindepth0 = readFromInput<real_t>(m_inputdata, "algorithm", "skindepth0");
  m_sigma0 = m_larmor0 * m_larmor0 / (m_skindepth0 * m_skindepth0);
  m_charge0 = 1.0 / (m_ppc0 * m_skindepth0 * m_skindepth0);
}

} // namespace ntt

// for (short i{dim}; i < 3; ++i) {
//   m_extent.push_back(0.0);
//   m_extent.push_back(0.0);
//   m_resolution.push_back(0);
// }

// // copy extent and resolution to device (or don't do anything if device ==
// host) NTTArray<real_t[6]>::HostMirror extent_host =
// Kokkos::create_mirror_view(extent); NTTArray<std::size_t[3]>::HostMirror
// resolution_host = Kokkos::create_mirror_view(resolution);
//
// for (short i{0}; i < 3; ++i) {
//   resolution_host(i) = m_resolution[i];
//   extent_host(2 * i) = m_extent[2 * i];
//   extent_host(2 * i + 1) = m_extent[2 * i + 1];
// }
//
// Kokkos::deep_copy(extent, extent_host);
// Kokkos::deep_copy(resolution, resolution_host);
