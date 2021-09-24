#include "global.h"
#include "sim.h"
#include "cargs.h"
#include "input.h"

#include <toml/toml.hpp>
#include <plog/Log.h>

#include <iostream>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <cassert>

namespace ntt {

Simulation::Simulation(Dimension dim, CoordinateSystem coord_sys, SimulationType sim_type)
    : m_simulation_type(sim_type), m_domain(dim, coord_sys) {
  // check compatibility
  if (((dim == ONE_D) && (coord_sys != CARTESIAN_COORD)) ||
      ((dim == TWO_D) && ((coord_sys == SPHERICAL_COORD) || (coord_sys == LOG_SPHERICAL_COORD))) ||
      ((dim == THREE_D) && (coord_sys == POLAR_COORD))) {
    PLOGF << "Incompatibility between the dimension [" << dim << "] and the coordinate system [" << coord_sys << "]";
    throw std::logic_error("#Error: incompatible simulation configurations.");
  }
}

void Simulation::parseInput(int argc, char *argv[]) {
  io::CommandLineArguments cl_args;
  cl_args.readCommandLineArguments(argc, argv);
  m_inputfilename = cl_args.getArgument("-input", DEF_input_filename);
  m_outputpath = cl_args.getArgument("-output", DEF_output_path);
  m_inputdata = toml::parse(static_cast<std::string>(m_inputfilename));

  m_title = io::readFromInput<std::string>(m_inputdata, "simulation", "title", "PIC_Sim");
  m_runtime = io::readFromInput<real_t>(m_inputdata, "simulation", "runtime");
  m_timestep = io::readFromInput<real_t>(m_inputdata, "algorithm", "timestep");

  auto resolution = io::readFromInput<std::vector<int>>(m_inputdata, "domain", "resolution");
  auto extent = io::readFromInput<std::vector<real_t>>(m_inputdata, "domain", "extent", {0.0, 1.0, 0.0, 1.0, 0.0, 1.0});

  m_domain.set_extent(extent);
  m_domain.set_resolution(resolution);

  auto boundaries = io::readFromInput<std::vector<std::string>>(m_inputdata, "domain", "boundaries", {"PERIODIC", "PERIODIC", "PERIODIC"});
  std::vector<BoundaryCondition> bcs;
  std::size_t b {0};
  for (auto & bc : boundaries) {
    if (bc == "PERIODIC") {
      bcs.push_back(PERIODIC_BC);
    } else if (bc == "OPEN") {
      bcs.push_back(OPEN_BC);
    } else {
      bcs.push_back(UNDEFINED_BC);
    }
    ++b;
    if (b >= m_domain.m_resolution.size()) {
      break;
    }
  }
  m_domain.set_boundaries(bcs);

  m_inputparsed = true;
}

void Simulation::printDetails(std::ostream &os) {
  assert(m_inputparsed);
  os << "- [Simulation details]\n";
  os << "   title: " << m_title << "\n";
  os << "   type: " << stringifySimulationType(m_simulation_type) << "\n";
  os << "   total runtime: " << m_runtime << "\n";
  os << "   dt: " << m_timestep << " [" << static_cast<int>(m_runtime / m_timestep) << "]\n";
  m_domain.printDetails(os);
}
void Simulation::printDetails() { printDetails(std::cout); }

void Simulation::initialize() {
  assert(m_inputparsed);
  m_initialized = true;
}
void Simulation::verify() {
  assert(is_inputparsed());
  assert(is_initialized());
  assert(get_simulation_type() != UNDEFINED_SIM);
  assert(get_dimension() != UNDEFINED_D);
  assert(get_coord_system() != UNDEFINED_COORD);
}
void Simulation::mainloop() {}
void Simulation::finalize() {
  assert(m_initialized);
  m_initialized = false;
}

auto Simulation::getSizeInBytes() -> std::size_t {
  return 0;
}

} // namespace ntt
