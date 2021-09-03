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
      PLOGW << "Cannot find variable <" << variable << "> from block [" << blockname << "] in the input file.";
      throw std::invalid_argument("Cannot find variable in input file.");
    }
  } else {
    PLOGW << "Cannot find block [" << blockname << "] in the input file.";
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
    PLOGW << "Variable <" << variable << "> of [" << blockname << "] not found. Falling back to default value.";
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
void Simulation::finalize() {
  assert(m_initialized);
  m_initialized = false;
}
void Simulation::mainloop() {
  assert(m_inputparsed);
  assert(m_initialized);
}

void PICSimulation::printDetails(std::ostream &os) {
  Simulation::printDetails(os);
  os << "   particle pusher: " << stringifyParticlePusher(m_pusher) << "\n";
}

void PICSimulation1D::initialize() {
  ex1.allocate(m_domain.nx1());
  ex2.allocate(m_domain.nx1());
  ex3.allocate(m_domain.nx1());
  bx1.allocate(m_domain.nx1());
  bx2.allocate(m_domain.nx1());
  bx3.allocate(m_domain.nx1());
  Simulation::initialize();
}

void PICSimulation2D::initialize() {
  ex1.allocate(m_domain.nx1(), m_domain.nx2());
  ex2.allocate(m_domain.nx1(), m_domain.nx2());
  ex3.allocate(m_domain.nx1(), m_domain.nx2());
  bx1.allocate(m_domain.nx1(), m_domain.nx2());
  bx2.allocate(m_domain.nx1(), m_domain.nx2());
  bx3.allocate(m_domain.nx1(), m_domain.nx2());
  Simulation::initialize();
}

void PICSimulation3D::initialize() {
  ex1.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  ex2.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  ex3.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  bx1.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  bx2.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  bx3.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  Simulation::initialize();
}

// explicitly calling all the destructors
void PICSimulation1D::finalize() {
  ex1.arrays::OneDArray<real_t>::~OneDArray<real_t>();
  ex2.arrays::OneDArray<real_t>::~OneDArray<real_t>();
  ex3.arrays::OneDArray<real_t>::~OneDArray<real_t>();
  bx1.arrays::OneDArray<real_t>::~OneDArray<real_t>();
  bx2.arrays::OneDArray<real_t>::~OneDArray<real_t>();
  bx3.arrays::OneDArray<real_t>::~OneDArray<real_t>();
  Simulation::finalize();
}

void PICSimulation2D::finalize() {
  ex1.arrays::TwoDArray<real_t>::~TwoDArray<real_t>();
  ex2.arrays::TwoDArray<real_t>::~TwoDArray<real_t>();
  ex3.arrays::TwoDArray<real_t>::~TwoDArray<real_t>();
  bx1.arrays::TwoDArray<real_t>::~TwoDArray<real_t>();
  bx2.arrays::TwoDArray<real_t>::~TwoDArray<real_t>();
  bx3.arrays::TwoDArray<real_t>::~TwoDArray<real_t>();
  Simulation::finalize();
}

void PICSimulation3D::finalize() {
  ex1.arrays::ThreeDArray<real_t>::~ThreeDArray<real_t>();
  ex2.arrays::ThreeDArray<real_t>::~ThreeDArray<real_t>();
  ex3.arrays::ThreeDArray<real_t>::~ThreeDArray<real_t>();
  bx1.arrays::ThreeDArray<real_t>::~ThreeDArray<real_t>();
  bx2.arrays::ThreeDArray<real_t>::~ThreeDArray<real_t>();
  bx3.arrays::ThreeDArray<real_t>::~ThreeDArray<real_t>();
  Simulation::finalize();
}

void PICSimulation::mainloop() {
  Simulation::mainloop();
  // ...
}

} // namespace ntt
