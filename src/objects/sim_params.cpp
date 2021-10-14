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

namespace ntt {

SimulationParams::SimulationParams(const toml::value& inputdata, Dimension dim) {
  m_inputdata = inputdata;

  m_title = readFromInput<std::string>(m_inputdata, "simulation", "title", "PIC_Sim");
  m_runtime = readFromInput<real_t>(m_inputdata, "simulation", "runtime");
  m_correction = readFromInput<real_t>(m_inputdata, "algorithm", "correction");

  auto nspec = readFromInput<int>(m_inputdata, "particles", "n_species");
  for (int i {0}; i < nspec; ++i) {
    auto label = readFromInput<std::string>(
        m_inputdata, "species_" + std::to_string(i + 1), "label", "s" + std::to_string(i + 1));
    auto mass = readFromInput<float>(m_inputdata, "species_" + std::to_string(i + 1), "mass");
    auto charge = readFromInput<float>(m_inputdata, "species_" + std::to_string(i + 1), "charge");
    auto maxnpart = static_cast<std::size_t>(
        readFromInput<double>(m_inputdata, "species_" + std::to_string(i + 1), "maxnpart"));
    auto pusher_str = readFromInput<std::string>(
        m_inputdata, "species_" + std::to_string(i + 1), "pusher", "Boris");
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

  // TODO: for now only PIC
  m_simtype = PIC_SIM;

  // TODO: hardcoded coord system
  m_coord_system = CARTESIAN_COORD;

  // auto coords = readFromInput<std::string>(m_inputdata, "domain", "coord_system", "XYZ");
  // if (coords == "X") {
  //   if (dim != ONE_D) { throw std::logic_error("ERROR: wrong coord system for given dimension.");
  //   } m_coord_system = CARTESIAN_COORD;
  // } else if (coords == "XY") {
  //   if (dim == THREE_D) {
  //     throw std::logic_error("ERROR: wrong coord system for given dimension.");
  //   }
  //   m_coord_system = CARTESIAN_COORD;
  // } else if (coords == "XYZ") {
  //   m_coord_system = CARTESIAN_COORD;
  // } else if (coords == "R_PHI") {
  //   if (dim != TWO_D) { throw std::logic_error("ERROR: wrong coord system for given dimension.");
  //   } m_coord_system = POLAR_R_PHI_COORD;
  // } else if (coords == "R_THETA") {
  //   if (dim != TWO_D) { throw std::logic_error("ERROR: wrong coord system for given dimension.");
  //   } m_coord_system = POLAR_R_THETA_COORD;
  // } else if (coords == "R_THETA_PHI") {
  //   if (dim != THREE_D) {
  //     throw std::logic_error("ERROR: wrong coord system for given dimension.");
  //   }
  //   m_coord_system = SPHERICAL_COORD;
  // } else if (coords == "logR_THETA_PHI") {
  //   if (dim != THREE_D) {
  //     throw std::logic_error("ERROR: wrong coord system for given dimension.");
  //   }
  //   m_coord_system = LOG_SPHERICAL_COORD;
  // } else {
  //   throw std::invalid_argument("Unknown coordinate system specified in the input.");
  // }

  // box size/resolution
  m_resolution = readFromInput<std::vector<std::size_t>>(m_inputdata, "domain", "resolution");
  m_extent = readFromInput<std::vector<real_t>>(
      m_inputdata, "domain", "extent", {0.0, 1.0, 0.0, 1.0, 0.0, 1.0});

  if ((static_cast<short>(m_resolution.size()) < static_cast<short>(dim))
      || (static_cast<short>(m_extent.size()) < 2 * static_cast<short>(dim))) {
    throw std::invalid_argument("Not enough values in `extent` or `resolution` input.");
  }

  m_resolution.erase(m_resolution.begin() + static_cast<short>(dim), m_resolution.end());
  m_extent.erase(m_extent.begin() + 2 * static_cast<short>(dim), m_extent.end());

  auto boundaries = readFromInput<std::vector<std::string>>(
      m_inputdata, "domain", "boundaries", {"PERIODIC", "PERIODIC", "PERIODIC"});
  short b {0};
  for (auto& bc : boundaries) {
    if (bc == "PERIODIC") {
      m_boundaries.push_back(PERIODIC_BC);
    } else if (bc == "OPEN") {
      m_boundaries.push_back(OPEN_BC);
    } else {
      m_boundaries.push_back(UNDEFINED_BC);
    }
    ++b;
    if (b >= static_cast<short>(dim)) { break; }
  }
  // plasma params
  m_ppc0 = readFromInput<real_t>(m_inputdata, "algorithm", "ppc0");
  m_larmor0 = readFromInput<real_t>(m_inputdata, "algorithm", "larmor0");
  m_skindepth0 = readFromInput<real_t>(m_inputdata, "algorithm", "skindepth0");
  m_sigma0 = m_larmor0 * m_larmor0 / (m_skindepth0 * m_skindepth0);
  m_charge0 = 1.0 / (m_ppc0 * m_skindepth0 * m_skindepth0);
  m_B0 = 1.0 / m_larmor0;

  // real_t maxtstep {}
  auto cfl = readFromInput<real_t>(m_inputdata, "algorithm", "CFL", 0.45);
  assert(cfl > 0);
  m_timestep = get_cell_size() * cfl;
}

auto SimulationParams::get_cell_size() -> real_t {
  // TODO: modify this for arbitrary coord system
  if (m_coord_system == CARTESIAN_COORD) {
    return (m_extent[1] - m_extent[0]) / static_cast<real_t>(m_resolution[0]);
  } else {
    throw std::logic_error("ERROR: Cannot determine the cell size.");
  }
}

void SimulationParams::verify() {
  if (m_simtype == UNDEFINED_SIM) { throw std::logic_error("ERROR: simulation type unspecified."); }
  if (m_coord_system == UNDEFINED_COORD) {
    throw std::logic_error("ERROR: coordinate system unspecified.");
  }
  for (auto& b : m_boundaries) {
    if (b == UNDEFINED_BC) { throw std::logic_error("ERROR: boundary conditions unspecified."); }
  }
  // TODO: maybe some other tests
}

void SimulationParams::printDetails() {
  PLOGI << "[Simulation details]";
  PLOGI << "   title: " << m_title;
  PLOGI << "   type: " << stringifySimulationType(m_simtype);
  PLOGI << "   total runtime: " << m_runtime;
  PLOGI << "   dt: " << m_timestep << " [" << static_cast<int>(m_runtime / m_timestep) << " steps]";

  auto dim = static_cast<short>(m_resolution.size());
  PLOGI << "[domain]";
  PLOGI << "   dimension: " << dim << "D";
  PLOGI << "   coordinate system: " << stringifyCoordinateSystem(m_coord_system, dim);

  std::string bc {"   boundary conditions: { "};
  for (auto& b : m_boundaries) {
    bc += stringifyBoundaryCondition(b) + " x ";
  }
  bc.erase(bc.size() - 3);
  bc += " }";
  PLOGI << bc;

  std::string res {"   resolution: { "};
  for (auto& r : m_resolution) {
    res += std::to_string(r) + " x ";
  }
  res.erase(res.size() - 3);
  res += " }";
  PLOGI << res;

  std::string ext {"   extent: "};
  for (std::size_t i {0}; i < m_extent.size(); i += 2) {
    ext += "{" + std::to_string(m_extent[i]) + ", " + std::to_string(m_extent[i + 1]) + "} ";
  }
  PLOGI << ext;

  std::string cell {"   cell size: "};
  real_t effective_dx {0.0};
  if (m_coord_system == CARTESIAN_COORD) {
    cell += "{" + std::to_string(get_cell_size()) + "}";
    effective_dx = get_cell_size();
  }
  PLOGI << cell;

  PLOGI << "[fiducial parameters]";
  PLOGI << "   ppc0: " << m_ppc0;
  PLOGI << "   rho0: " << m_larmor0 << " [" << m_larmor0 / effective_dx << " dx]";
  PLOGI << "   c_omp0: " << m_skindepth0 << " [" << m_skindepth0 / effective_dx << " dx]";
  PLOGI << "   sigma0: " << m_sigma0;
  PLOGI << "   q0: " << m_charge0;
  PLOGI << "   B0: " << m_B0;
}

} // namespace ntt
