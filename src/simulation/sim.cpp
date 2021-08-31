#include "global.h"
#include "sim.h"
#include "cargs.h"
#include "input.h"

#include <toml/toml.hpp>
#include <plog/Log.h>

#include <iostream>
#include <exception>

namespace ntt {
Simulation::Simulation(Dimension dim, CoordinateSystem coord_sys, SimulationType sim_type)
    : m_dimension(dim), m_coord_system(coord_sys), m_simulation_type(sim_type) {
  // check compatibility
  try {
    if (((m_dimension == ONE_D) && (m_coord_system != CARTESIAN_COORD)) ||
        ((m_dimension == TWO_D) && ((m_coord_system == SPHERICAL_COORD) ||
                                    (m_coord_system == LOG_SPHERICAL_COORD))) ||
        ((m_dimension == THREE_D) && (m_coord_system == POLAR_COORD))) {
      PLOGF << "Incompatibility between the dimension [" << m_dimension
            << "] and the coordinate system [" << m_coord_system << "]";
      throw std::logic_error("");
    }
  } catch (std::exception &err) {
    std::cerr << "Incompatible dimension & coordinate system." << std::endl;
    return;
  }
}

template<typename T>
T Simulation::readFromInput(const std::string &blockname, const std::string &variable) {
  try {
    io::dataExistsInToml(m_inputdata, blockname, variable);
  } catch (std::exception &err) {
    std::cerr << err.what() << std::endl;
  }
  return io::readTomlData<T>(m_inputdata, blockname, variable);
}

void Simulation::parseInput(int argc, char *argv[]) {
  io::CommandLineArguments cl_args;
  cl_args.readCommandLineArguments(argc, argv);
  m_inputdata = toml::parse(static_cast<std::string>(cl_args.getArgument("-input", DEF_input_filename)));
}
} // namespace ntt
