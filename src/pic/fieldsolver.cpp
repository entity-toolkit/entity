#include "global.h"
#include "picsim.h"

#include <iostream>

namespace ntt {

void PICSimulation1D::faradayHalfsubstep(const real_t &time) {
  std::cout << stringifyDimension(m_domain.m_dimension) << " faraday " << time << "\n";
}

void PICSimulation2D::faradayHalfsubstep(const real_t &time) {
  std::cout << stringifyDimension(m_domain.m_dimension) << " faraday " << time << "\n";
}

void PICSimulation3D::faradayHalfsubstep(const real_t &time) {
  std::cout << stringifyDimension(m_domain.m_dimension) << " faraday " << time << "\n";
}

void PICSimulation1D::ampereSubstep(const real_t &time) {
  std::cout << stringifyDimension(m_domain.m_dimension) << " ampere " << time << "\n";
}

void PICSimulation2D::ampereSubstep(const real_t &time) {
  std::cout << stringifyDimension(m_domain.m_dimension) << " ampere " << time << "\n";
}

void PICSimulation3D::ampereSubstep(const real_t &time) {
  std::cout << stringifyDimension(m_domain.m_dimension) << " ampere " << time << "\n";
}

void PICSimulation1D::addCurrentsSubstep(const real_t &time) {
  std::cout << stringifyDimension(m_domain.m_dimension) << " currents " << time << "\n";
}

void PICSimulation2D::addCurrentsSubstep(const real_t &time) {
  std::cout << stringifyDimension(m_domain.m_dimension) << " currents " << time << "\n";
}

void PICSimulation3D::addCurrentsSubstep(const real_t &time) {
  std::cout << stringifyDimension(m_domain.m_dimension) << " currents " << time << "\n";
}


}
