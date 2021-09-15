#include "global.h"
#include "timer.h"
#include "sim.h"
#include "picsim.h"
#include "input.h"

#include <iostream>
#include <cstddef>

namespace ntt {

void PICSimulation::printDetails(std::ostream &os) {
  Simulation::printDetails(os);
  os << ". [particles]\n";
  int s {1};
  for (auto & spec : m_species) {
    spec.printDetails(os, s);
    ++s;
  }
}

void PICSimulation::parseInput(int argc, char *argv[]) {
  Simulation::parseInput(argc, argv);
  auto nspec = io::readFromInput<int>(m_inputdata, "particles", "n_species");
  for (int i{0}; i < nspec; ++i) {
    m_species.emplace_back();
    // each species knows dimensionality
    m_species[i].setDimension(m_domain.m_dimension);
    // each species knows its mass & charge
    auto mass = io::readFromInput<float>(m_inputdata, "species_" + std::to_string(i + 1), "mass");
    auto charge = io::readFromInput<float>(m_inputdata, "species_" + std::to_string(i + 1), "charge");
    m_species[i].setMass(mass);
    m_species[i].setCharge(charge);
    // each species knows its max allowed particle number
    auto maxnpart = static_cast<std::size_t>(io::readFromInput<double>(m_inputdata, "species_" + std::to_string(i + 1), "maxnpart"));
    m_species[i].setMaxnpart(maxnpart);
  }
}

void PICSimulation::initialize() {
  Simulation::initialize();
  for (auto & spec : m_species) {
    spec.allocate();
  }
}
void PICSimulation::verify() {
  Simulation::verify();
  // TODO
  // 1. check that boundaries are assigned (verify they're consistent with dimensionality and coord system)
  // 2. check that fields are allocated
  // 3. check that particles are allocated or maxnpart == 0
  // 4. check that the dimensionality/coord system of particles is the same as for the domain
  // 5. check that particles have a pusher
}

void TEST_printReport(const timer::Timer &tm) {
  tm.printElapsed(timer::nanosecond);
}

void PICSimulation::mainloop() {
  for (real_t time {0}; time < m_runtime; time += m_timestep) {
    stepForward(time);
    TEST_printReport(timer_em);
    TEST_printReport(timer_pusher);
    TEST_printReport(timer_deposit);
  }
}

auto PICSimulation::getSizeInBytes() -> std::size_t {
  std::size_t size_in_bytes {0};
  for (auto spec : m_species) {
    size_in_bytes += spec.getSizeInBytes();
  }
  return Simulation::getSizeInBytes() + size_in_bytes;
}

void PICSimulation1D::initialize() {
  ex1.allocate(m_domain.nx1());
  ex2.allocate(m_domain.nx1());
  ex3.allocate(m_domain.nx1());
  bx1.allocate(m_domain.nx1());
  bx2.allocate(m_domain.nx1());
  bx3.allocate(m_domain.nx1());
  jx1.allocate(m_domain.nx1());
  jx2.allocate(m_domain.nx1());
  jx3.allocate(m_domain.nx1());
  PICSimulation::initialize();
}

void PICSimulation2D::initialize() {
  ex1.allocate(m_domain.nx1(), m_domain.nx2());
  ex2.allocate(m_domain.nx1(), m_domain.nx2());
  ex3.allocate(m_domain.nx1(), m_domain.nx2());
  bx1.allocate(m_domain.nx1(), m_domain.nx2());
  bx2.allocate(m_domain.nx1(), m_domain.nx2());
  bx3.allocate(m_domain.nx1(), m_domain.nx2());
  jx1.allocate(m_domain.nx1(), m_domain.nx2());
  jx2.allocate(m_domain.nx1(), m_domain.nx2());
  jx3.allocate(m_domain.nx1(), m_domain.nx2());
  PICSimulation::initialize();
}

void PICSimulation3D::initialize() {
  ex1.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  ex2.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  ex3.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  bx1.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  bx2.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  bx3.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  jx1.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  jx2.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  jx3.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  PICSimulation::initialize();
}

void PICSimulation1D::verify() {
  PICSimulation::verify();
}
void PICSimulation2D::verify() {
  PICSimulation::verify();
}
void PICSimulation3D::verify() {
  PICSimulation::verify();
}

auto PICSimulation1D::getSizeInBytes() -> std::size_t {
  std::size_t size_in_bytes {0};
  size_in_bytes += ex1.getSizeInBytes();
  size_in_bytes += ex2.getSizeInBytes();
  size_in_bytes += ex3.getSizeInBytes();
  size_in_bytes += bx1.getSizeInBytes();
  size_in_bytes += bx2.getSizeInBytes();
  size_in_bytes += bx3.getSizeInBytes();
  size_in_bytes += jx1.getSizeInBytes();
  size_in_bytes += jx2.getSizeInBytes();
  size_in_bytes += jx3.getSizeInBytes();
  return PICSimulation::getSizeInBytes() + size_in_bytes;
}

auto PICSimulation2D::getSizeInBytes() -> std::size_t {
  std::size_t size_in_bytes {0};
  size_in_bytes += ex1.getSizeInBytes();
  size_in_bytes += ex2.getSizeInBytes();
  size_in_bytes += ex3.getSizeInBytes();
  size_in_bytes += bx1.getSizeInBytes();
  size_in_bytes += bx2.getSizeInBytes();
  size_in_bytes += bx3.getSizeInBytes();
  size_in_bytes += jx1.getSizeInBytes();
  size_in_bytes += jx2.getSizeInBytes();
  size_in_bytes += jx3.getSizeInBytes();
  return PICSimulation::getSizeInBytes() + size_in_bytes;
}

auto PICSimulation3D::getSizeInBytes() -> std::size_t {
  std::size_t size_in_bytes {0};
  size_in_bytes += ex1.getSizeInBytes();
  size_in_bytes += ex2.getSizeInBytes();
  size_in_bytes += ex3.getSizeInBytes();
  size_in_bytes += bx1.getSizeInBytes();
  size_in_bytes += bx2.getSizeInBytes();
  size_in_bytes += bx3.getSizeInBytes();
  size_in_bytes += jx1.getSizeInBytes();
  size_in_bytes += jx2.getSizeInBytes();
  size_in_bytes += jx3.getSizeInBytes();
  return PICSimulation::getSizeInBytes() + size_in_bytes;
}

} // namespace ntt
