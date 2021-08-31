#include "global.h"
#include "sim.h"
#include "cargs.h"
#include "input.h"

#include <toml/toml.hpp>
#include <plog/Log.h>

#include <iostream>
#include <stdexcept>

namespace ntt {
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

template <typename T> auto Simulation::readFromInput(const std::string &blockname, const std::string &variable) -> T {
  return io::readTomlData<T>(m_inputdata, blockname, variable);
}

void Simulation::parseInput(int argc, char *argv[]) {
  io::CommandLineArguments cl_args;
  cl_args.readCommandLineArguments(argc, argv);
  m_inputfilename = cl_args.getArgument("-input", DEF_input_filename);
  m_outputpath = cl_args.getArgument("-output", DEF_output_path);
  m_inputdata = toml::parse(static_cast<std::string>(m_inputfilename));
}
} // namespace ntt
