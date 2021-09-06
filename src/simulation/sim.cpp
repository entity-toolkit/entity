#include "global.h"
#include "sim.h"
#include "cargs.h"

#include <toml/toml.hpp>
#include <plog/Log.h>

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>

namespace ntt {
namespace { // anonymous namespace
void dataExistsInToml(toml::value inputdata, const std::string &blockname, const std::string &variable) {
  if (inputdata.contains(blockname)) {
    auto &val_block = toml::find(inputdata, blockname);
    if (!val_block.contains(variable)) {
      PLOGI << "Cannot find variable <" << variable << "> from block [" << blockname << "] in the input file.";
      throw std::invalid_argument("Cannot find variable in input file.");
    }
  } else {
    PLOGI << "Cannot find block [" << blockname << "] in the input file.";
    throw std::invalid_argument("Cannot find blockname in input file.");
  }
}
} // namespace

template <typename T> auto Simulation::readFromInput(const std::string &blockname, const std::string &variable) -> T {
  dataExistsInToml(m_inputdata, blockname, variable);
  auto &val_block = toml::find(m_inputdata, blockname);
  return toml::find<T>(val_block, variable);
}
template <typename T>
auto Simulation::readFromInput(const std::string &blockname, const std::string &variable, const T &defval) -> T {
  try {
    return readFromInput<T>(blockname, variable);
  } catch (std::exception &err) {
    PLOGI << "Variable <" << variable << "> of [" << blockname << "] not found. Falling back to default value.";
    return defval;
  }
}

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

  m_title = readFromInput<std::string>("simulation", "title", "PIC_Sim");
  m_runtime = readFromInput<real_t>("simulation", "runtime");
  m_timestep = readFromInput<real_t>("algorithm", "timestep");

  auto resolution = readFromInput<std::vector<int>>("domain", "resolution");
  auto extent = readFromInput<std::vector<real_t>>("domain", "extent", {0.0, 1.0, 0.0, 1.0, 0.0, 1.0});

  m_domain.set_extent(extent);
  m_domain.set_resolution(resolution);
  // TODO: update default boundaries
  m_inputparsed = true;
}

void Simulation::printDetails(std::ostream &os) {
  assert(m_inputparsed);
  os << "[Simulation details]\n";
  os << "Title: " << m_title << "\n";
  os << "   type: " << stringifySimulationType(m_simulation_type) << "\n";
  os << "   total runtime: " << m_runtime << "\n";
  os << "   dt: " << m_timestep << " [" << static_cast<int>(m_runtime / m_timestep) << "]\n";
  os << "   resolution: ";
  for (auto r : get_resolution()) {
    os << r << " x ";
  }
  os << "\b\b  \n";
  os << "   size: ";
  auto extent = get_extent();
  for (std::size_t i{0}; i < extent.size(); i += 2) {
    os << "[" << extent[i] << ", " << extent[i + 1] << "] ";
  }
  os << "\n";
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

} // namespace ntt
