#include "global.h"
#include "sim.h"
#include "cargs.h"

#include <toml/toml.hpp>
#include <plog/Log.h>

#include <iostream>
#include <vector>
#include <stdexcept>

namespace ntt {
namespace { // anonymous namespace
  void dataExistsInToml(toml::value inputdata, const std::string &blockname, const std::string &variable) {
    if (inputdata.contains(blockname)) {
      auto &val_block = toml::find(inputdata, blockname);
      if (!val_block.contains(variable)) {
        PLOGW << "Cannot find variable <" << variable << "> from block [" << blockname << "] in the input file.";
        throw std::invalid_argument("Cannot find variable in input file.");
      }
    } else {
      PLOGW << "Cannot find block [" << blockname << "] in the input file.";
      throw std::invalid_argument("Cannot find blockname in input file.");
    }
  }
}

Simulation::Simulation(Dimension dim, CoordinateSystem coord_sys, SimulationType sim_type)
    : m_dimension(dim), m_coord_system(coord_sys), m_simulation_type(sim_type) {
  // check compatibility
  if (((m_dimension == ONE_D) && (m_coord_system != CARTESIAN_COORD)) ||
      ((m_dimension == TWO_D) && ((m_coord_system == SPHERICAL_COORD) || (m_coord_system == LOG_SPHERICAL_COORD))) ||
      ((m_dimension == THREE_D) && (m_coord_system == POLAR_COORD))) {
    PLOGF << "Incompatibility between the dimension [" << m_dimension << "] and the coordinate system ["
          << m_coord_system << "]";
    throw std::logic_error("#Error: incompatible simulation configurations.");
  }
}

void Simulation::parseInput(int argc, char *argv[]) {
  io::CommandLineArguments cl_args;
  cl_args.readCommandLineArguments(argc, argv);
  m_inputfilename = cl_args.getArgument("-input", DEF_input_filename);
  m_outputpath = cl_args.getArgument("-output", DEF_output_path);
  m_inputdata = toml::parse(static_cast<std::string>(m_inputfilename));

  m_title = readFromInput<std::string>("simulation", "title", "PIC_Sim");
  m_runtime = readFromInput<real_t>("simulation", "runtime");
  m_resolution = readFromInput<std::vector<int>>("domain", "resolution");
  m_size = readFromInput<std::vector<real_t>>("domain", "size", {0.0, 1.0, 0.0, 1.0, 0.0, 1.0});

  m_timestep = readFromInput<real_t>("algorithm", "timestep");

  // TODO: define the domain object here
  // define converter functions (cells) -> (coordinates)

  // check that everything is defined consistently
  if (m_dimension == ONE_D) {
    if (m_resolution.size() > 1) {
      PLOGW << "1D simulation specified, ignoring extra dimensions in `resolution`.";
      m_resolution.erase(m_resolution.begin() + 1, m_resolution.end());
    }
    if (m_size.size() > 2) {
      PLOGW << "1D simulation specified, ignoring extra dimensions in `size`.";
      m_size.erase(m_size.begin() + 2, m_size.end());
    }
  } else if (m_dimension == TWO_D) {
    if (m_resolution.size() > 2) {
      PLOGW << "2D simulation specified, ignoring extra dimensions in `resolution`.";
      m_resolution.erase(m_resolution.begin() + 2, m_resolution.end());
    } else if (m_resolution.size() < 2) {
      PLOGE << "2D simulation specified, not enough dimensions given in the input.";
      throw std::invalid_argument("Not enough values in `resolution` input.");
    }
    if (m_size.size() > 4) {
      PLOGW << "2D simulation specified, ignoring extra dimensions in `size`.";
      m_size.erase(m_size.begin() + 4, m_size.end());
    } else if (m_size.size() < 4) {
      PLOGE << "2D simulation specified, not enough dimensions given in the input.";
      throw std::invalid_argument("Not enough values in `size` input.");
    }
  } else if (m_dimension == THREE_D) {
    if (m_resolution.size() > 3) {
      PLOGW << "3D simulation specified, ignoring extra dimensions in `resolution`.";
      m_resolution.erase(m_resolution.begin() + 3, m_resolution.end());
    } else if (m_resolution.size() < 3) {
      PLOGE << "3D simulation specified, not enough dimensions given in the input.";
      throw std::invalid_argument("Not enough values in `resolution` input.");
    }
    if (m_size.size() > 6) {
      PLOGW << "3D simulation specified, ignoring extra dimensions in `size`.";
      m_size.erase(m_size.begin() + 6, m_size.end());
    } else if (m_size.size() < 6) {
      PLOGE << "3D simulation specified, not enough dimensions given in the input.";
      throw std::invalid_argument("Not enough values in `size` input.");
    }
  } else {
    throw std::runtime_error("# Error: unknown dimension of simulation.");
  }
}

void Simulation::printDetails(std::ostream& os) {
  os << "[Simulation details]\n";
  os << "Title: " << m_title << "\n";
  os << "   type: " << stringifySimulationType(m_simulation_type) << "\n";
  os << "   dim: " << stringifyDimension(m_dimension) << "\n";
  os << "   coord: " << stringifyCoordinateSystem(m_coord_system) << "\n\n";
  os << "   total runtime: " << m_runtime << "\n";
  os << "   dt: " << m_timestep << " [" << static_cast<int>(m_runtime / m_timestep) << "]\n";
  os << "   resolution: ";
  for (auto r: m_resolution) {
    os << r << " x ";
  }
  os << "\b\b  \n";
  os << "   size: ";
  for(std::size_t i {0}; i < m_size.size(); i += 2) {
    os << "[" << m_size[i] << ", " << m_size[i + 1] << "] ";
  }
  os << "\n";
}
void Simulation::printDetails() {
  printDetails(std::cout);
}

template <typename T>
auto Simulation::readFromInput(const std::string &blockname, const std::string &variable) -> T {
  dataExistsInToml(m_inputdata, blockname, variable);
  auto &val_block = toml::find(m_inputdata, blockname);
  return toml::find<T>(val_block, variable);
}
template <typename T>
auto Simulation::readFromInput(const std::string &blockname, const std::string &variable, const T &defval) -> T {
  try {
    return readFromInput<T>(blockname, variable);
  } catch (std::exception &err) {
    return defval;
  }
}

void Simulation::initialize() {
  m_initialized = true;
}
void Simulation::finalize() {
  m_initialized = false;
}
void Simulation::mainloop() {}

void PICSimulation::printDetails(std::ostream& os) {
  Simulation::printDetails(os);
  os << "   particle pusher: " << stringifyParticlePusher(m_pusher) << "\n";
}

void PICSimulation1D::initialize() {
  ex1.allocate(m_resolution[0]);
  ex2.allocate(m_resolution[0]);
  ex3.allocate(m_resolution[0]);
  bx1.allocate(m_resolution[0]);
  bx2.allocate(m_resolution[0]);
  bx3.allocate(m_resolution[0]);
  Simulation::initialize();
}

void PICSimulation2D::initialize() {
  ex1.allocate(m_resolution[0], m_resolution[1]);
  ex2.allocate(m_resolution[0], m_resolution[1]);
  ex3.allocate(m_resolution[0], m_resolution[1]);
  bx1.allocate(m_resolution[0], m_resolution[1]);
  bx2.allocate(m_resolution[0], m_resolution[1]);
  bx3.allocate(m_resolution[0], m_resolution[1]);
  Simulation::initialize();
}

void PICSimulation3D::initialize() {
  ex1.allocate(m_resolution[0], m_resolution[1], m_resolution[2]);
  ex2.allocate(m_resolution[0], m_resolution[1], m_resolution[2]);
  ex3.allocate(m_resolution[0], m_resolution[1], m_resolution[2]);
  bx1.allocate(m_resolution[0], m_resolution[1], m_resolution[2]);
  bx2.allocate(m_resolution[0], m_resolution[1], m_resolution[2]);
  bx3.allocate(m_resolution[0], m_resolution[1], m_resolution[2]);
  Simulation::initialize();
}

void PICSimulation1D::finalize() {
  ex1.~OneDArray<real_t>(); ex2.~OneDArray<real_t>(); ex3.~OneDArray<real_t>();
  bx1.~OneDArray<real_t>(); bx2.~OneDArray<real_t>(); bx3.~OneDArray<real_t>();
  Simulation::finalize();
}

void PICSimulation2D::finalize() {
  ex1.~TwoDArray<real_t>(); ex2.~TwoDArray<real_t>(); ex3.~TwoDArray<real_t>();
  bx1.~TwoDArray<real_t>(); bx2.~TwoDArray<real_t>(); bx3.~TwoDArray<real_t>();
  Simulation::finalize();
}

void PICSimulation3D::finalize() {
  ex1.~ThreeDArray<real_t>(); ex2.~ThreeDArray<real_t>(); ex3.~ThreeDArray<real_t>();
  bx1.~ThreeDArray<real_t>(); bx2.~ThreeDArray<real_t>(); bx3.~ThreeDArray<real_t>();
  Simulation::finalize();
}

} // namespace ntt
