#include "global.h"
#include "sim.h"
#include "cargs.h"

#include <iostream>
#include <cassert>

namespace ntt {
void PICSimulation::printDetails(std::ostream &os) {
  Simulation::printDetails(os);
  os << "   particle pusher: " << stringifyParticlePusher(m_pusher) << "\n";
}

void PICSimulation::initialize() {
  Simulation::initialize();
}
void PICSimulation::verify() {
  Simulation::verify();
  assert(get_particle_pusher() != UNDEFINED_PUSHER);
}
void PICSimulation::finalize() {
  Simulation::finalize();
}

void PICSimulation::mainloop() {
  Simulation::mainloop();
  // ...
}

void PICSimulation1D::initialize() {
  ex1.allocate(m_domain.nx1());
  ex2.allocate(m_domain.nx1());
  ex3.allocate(m_domain.nx1());
  bx1.allocate(m_domain.nx1());
  bx2.allocate(m_domain.nx1());
  bx3.allocate(m_domain.nx1());
  PICSimulation::initialize();
}

void PICSimulation2D::initialize() {
  ex1.allocate(m_domain.nx1(), m_domain.nx2());
  ex2.allocate(m_domain.nx1(), m_domain.nx2());
  ex3.allocate(m_domain.nx1(), m_domain.nx2());
  bx1.allocate(m_domain.nx1(), m_domain.nx2());
  bx2.allocate(m_domain.nx1(), m_domain.nx2());
  bx3.allocate(m_domain.nx1(), m_domain.nx2());
  PICSimulation::initialize();
}

void PICSimulation3D::initialize() {
  ex1.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  ex2.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  ex3.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  bx1.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  bx2.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
  bx3.allocate(m_domain.nx1(), m_domain.nx2(), m_domain.nx3());
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

// explicitly calling all the destructors
void PICSimulation1D::finalize() {
  ex1.arrays::OneDArray<real_t>::~OneDArray<real_t>();
  ex2.arrays::OneDArray<real_t>::~OneDArray<real_t>();
  ex3.arrays::OneDArray<real_t>::~OneDArray<real_t>();
  bx1.arrays::OneDArray<real_t>::~OneDArray<real_t>();
  bx2.arrays::OneDArray<real_t>::~OneDArray<real_t>();
  bx3.arrays::OneDArray<real_t>::~OneDArray<real_t>();
  PICSimulation::finalize();
}

void PICSimulation2D::finalize() {
  ex1.arrays::TwoDArray<real_t>::~TwoDArray<real_t>();
  ex2.arrays::TwoDArray<real_t>::~TwoDArray<real_t>();
  ex3.arrays::TwoDArray<real_t>::~TwoDArray<real_t>();
  bx1.arrays::TwoDArray<real_t>::~TwoDArray<real_t>();
  bx2.arrays::TwoDArray<real_t>::~TwoDArray<real_t>();
  bx3.arrays::TwoDArray<real_t>::~TwoDArray<real_t>();
  PICSimulation::finalize();
}

void PICSimulation3D::finalize() {
  ex1.arrays::ThreeDArray<real_t>::~ThreeDArray<real_t>();
  ex2.arrays::ThreeDArray<real_t>::~ThreeDArray<real_t>();
  ex3.arrays::ThreeDArray<real_t>::~ThreeDArray<real_t>();
  bx1.arrays::ThreeDArray<real_t>::~ThreeDArray<real_t>();
  bx2.arrays::ThreeDArray<real_t>::~ThreeDArray<real_t>();
  bx3.arrays::ThreeDArray<real_t>::~ThreeDArray<real_t>();
  PICSimulation::finalize();
}

} // namespace ntt
