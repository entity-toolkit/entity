#include "global.h"
#include "picsim.h"

#include <iostream>

namespace ntt {

void PICSimulation1D::depositSubstep(const real_t &time) {
  std::cout << stringifyDimension(m_domain.m_dimension) << " deposit " << time << "\n";
}

void PICSimulation2D::depositSubstep(const real_t &time) {
  std::cout << stringifyDimension(m_domain.m_dimension) << " deposit " << time << "\n";
}

void PICSimulation3D::depositSubstep(const real_t &time) {
  std::cout << stringifyDimension(m_domain.m_dimension) << " deposit " << time << "\n";
}

}
