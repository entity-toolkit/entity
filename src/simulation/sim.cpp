#include "sim.h"
#include "cargs.h"

#include <plog/Log.h>

#include <iostream>
#include <exception>

namespace ntt {
Simulation::Simulation(Dimension dim, CoordinateSystem coord_sys)
    : m_dimension(dim), m_coord_system(coord_sys) {
  // check compatibility
  try {
    if (((m_dimension == ONE_D) && (m_coord_system != CARTESIAN)) ||
        ((m_dimension == TWO_D) && ((m_coord_system == SPHERICAL) ||
                                    (m_coord_system == LOG_SPHERICAL))) ||
        ((m_dimension == THREE_D) && (m_coord_system == POLAR))) {
      PLOGF << "Incompatibility between the dimension [" << m_dimension
            << "] and the coordinate system [" << m_coord_system << "]";
      throw std::invalid_argument("");
    }
  } catch (std::exception &err) {
    std::cerr << "Invalid argument error." << std::endl;
    return;
  }
}

void Simulation::parseInput(int argc, char *argv[]) {
  io::CommandLineArguments cl_args;
  cl_args.readCommandLineArguments(argc, argv);
  m_input_params.set_input_filename(
      cl_args.getArgument("-input", DEF_input_filename));
  // m_input_params.set_parameter("one", "two", true);
  // (*m_input_params.get_parameter("one", "two")->value_bool);
}
} // namespace ntt
