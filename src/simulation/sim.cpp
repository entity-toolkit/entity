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
    dataExistsInToml(m_inputdata, blockname, variable);
    auto &val_block = toml::find(m_inputdata, blockname);
    return toml::find<T>(val_block, variable);
  } catch (std::exception &err) {
    return defval;
  }
}

void Simulation::initialize() {
  m_title = readFromInput<std::string>("simulation", "title", "PIC_Sim");
  m_runtime = readFromInput<real_t>("simulation", "runtime");

  m_resolution = readFromInput<std::vector<int>>("domain", "resolution");
  m_dimensions = readFromInput<std::vector<real_t>>("domain", "dimensions", {0.0, 1.0, 0.0, 1.0, 0.0, 1.0});

  m_timestep = readFromInput<real_t>("algorithm", "timestep");

  m_initialized = true;
}

void Simulation::printDetails(std::ostream& os) {
  os << "init: " << m_initialized << "\n";
  os << "[Simulation details]\n";
  os << "Title: " << m_title << "\n";
  os << "dt: " << m_timestep << "\n";
}
void Simulation::printDetails() {
  printDetails(std::cout);
}

} // namespace ntt
