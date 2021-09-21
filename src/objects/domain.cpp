#include "global.h"
#include "domain.h"

#include <plog/Log.h>

#include <cassert>
#include <vector>
#include <iostream>

namespace ntt {

void Domain::printDetails(std::ostream &os) {
  os << ". [domain]\n";
  os << "   dimension: " << stringifyDimension(m_dimension) << "\n";
  os << "   coordinate system: " << stringifyCoordinateSystem(m_coord_system) << "\n";
  os << "   boundary conditions: ";
  for (auto & b : m_boundary) {
    os << stringifyBoundaryCondition(b) << " x ";
  }
  os << "\b\b  \n";
  os << "   resolution: ";
  for (auto & r : m_resolution) {
    os << r << " x ";
  }
  os << "\b\b  \n";
  os << "   extent: ";
  auto extent = m_extent;
  for (std::size_t i{0}; i < extent.size(); i += 2) {
    os << "[" << extent[i] << ", " << extent[i + 1] << "] ";
  }
  os << "\n";
}
void Domain::printDetails() { printDetails(std::cout); }


void Domain::set_extent(const std::vector<real_t> &extent) {
  // check that everything is defined consistently
  m_extent = extent;
  if (m_dimension == ONE_D) {
    if (m_extent.size() > 2) {
      PLOGI << "1D simulation specified, ignoring extra dimensions in `extent`.";
      m_extent.erase(m_extent.begin() + 2, m_extent.end());
    }
  } else if (m_dimension == TWO_D) {
    if (m_extent.size() > 4) {
      PLOGI << "2D simulation specified, ignoring extra dimensions in `extent`.";
      m_extent.erase(m_extent.begin() + 4, m_extent.end());
    } else if (m_extent.size() < 4) {
      PLOGF << "2D simulation specified, not enough dimensions given in the input.";
      throw std::invalid_argument("Not enough values in `extent` input.");
    }
  } else if (m_dimension == THREE_D) {
    if (m_extent.size() > 6) {
      PLOGI << "3D simulation specified, ignoring extra dimensions in `extent`.";
      m_extent.erase(m_extent.begin() + 6, m_extent.end());
    } else if (m_extent.size() < 6) {
      PLOGF << "3D simulation specified, not enough dimensions given in the input.";
      throw std::invalid_argument("Not enough values in `extent` input.");
    }
  } else {
    throw std::runtime_error("# Error: unknown dimension of simulation.");
  }
}
void Domain::set_resolution(const std::vector<int> &resolution) {
  // check that everything is defined consistently
  m_resolution = resolution;
  if (m_dimension == ONE_D) {
    if (m_resolution.size() > 1) {
      PLOGI << "1D simulation specified, ignoring extra dimensions in `resolution`.";
      m_resolution.erase(m_resolution.begin() + 1, m_resolution.end());
    }
  } else if (m_dimension == TWO_D) {
    if (m_resolution.size() > 2) {
      PLOGI << "2D simulation specified, ignoring extra dimensions in `resolution`.";
      m_resolution.erase(m_resolution.begin() + 2, m_resolution.end());
    } else if (m_resolution.size() < 2) {
      PLOGF << "2D simulation specified, not enough dimensions given in the input.";
      throw std::invalid_argument("Not enough values in `resolution` input.");
    }
  } else if (m_dimension == THREE_D) {
    if (m_resolution.size() > 3) {
      PLOGI << "3D simulation specified, ignoring extra dimensions in `resolution`.";
      m_resolution.erase(m_resolution.begin() + 3, m_resolution.end());
    } else if (m_resolution.size() < 3) {
      PLOGF << "3D simulation specified, not enough dimensions given in the input.";
      throw std::invalid_argument("Not enough values in `resolution` input.");
    }
  } else {
    throw std::runtime_error("# Error: unknown dimension of simulation.");
  }
}

void Domain::set_boundary_x1(const BoundaryCondition &bc) {
  assert(m_dimension != UNDEFINED_D);
  assert(m_boundary.size() >= 1);
  m_boundary[0] = bc;
}
void Domain::set_boundary_x2(const BoundaryCondition &bc) {
  assert((m_dimension != UNDEFINED_D) && (m_dimension != ONE_D));
  assert(m_boundary.size() >= 2);
  m_boundary[1] = bc;
}
void Domain::set_boundary_x3(const BoundaryCondition &bc) {
  assert(m_dimension == THREE_D);
  assert(m_boundary.size() == 3);
  m_boundary[2] = bc;
}
void Domain::set_boundaries(const std::vector<BoundaryCondition> &bc) {
  assert(bc.size() == m_resolution.size());
  m_boundary = bc;
}

auto Domain::x1min() const -> real_t {
  assert((m_extent.size() >= 2) && (m_dimension != UNDEFINED_D));
  return m_extent[0];
}
auto Domain::x1max() const -> real_t {
  assert((m_extent.size() >= 2) && (m_dimension != UNDEFINED_D));
  return m_extent[1];
}
auto Domain::x2min() const -> real_t {
  assert((m_extent.size() >= 4) && (m_dimension != UNDEFINED_D) && (m_dimension != ONE_D));
  return m_extent[2];
}
auto Domain::x2max() const -> real_t {
  assert((m_extent.size() >= 4) && (m_dimension != UNDEFINED_D) && (m_dimension != ONE_D));
  return m_extent[3];
}
auto Domain::x3min() const -> real_t {
  assert((m_extent.size() == 6) && (m_dimension != UNDEFINED_D) && (m_dimension != ONE_D) && (m_dimension != TWO_D));
  return m_extent[4];
}
auto Domain::x3max() const -> real_t {
  assert((m_extent.size() == 6) && (m_dimension != UNDEFINED_D) && (m_dimension != ONE_D) && (m_dimension != TWO_D));
  return m_extent[5];
}
auto Domain::nx1() const -> int {
  assert((m_resolution.size() >= 1) && (m_dimension != UNDEFINED_D));
  return m_resolution[0];
}
auto Domain::nx2() const -> int {
  assert((m_resolution.size() >= 2) && (m_dimension != UNDEFINED_D) && (m_dimension != ONE_D));
  return m_resolution[1];
}
auto Domain::nx3() const -> int {
  assert((m_resolution.size() == 3) && (m_dimension != UNDEFINED_D) && (m_dimension != ONE_D) &&
         (m_dimension != TWO_D));
  return m_resolution[2];
}
auto Domain::sizex1() const -> real_t { return x1max() - x1min(); }
auto Domain::sizex2() const -> real_t { return x2max() - x2min(); }
auto Domain::sizex3() const -> real_t { return x3max() - x3min(); }

auto Domain::sizexi() const -> std::vector<real_t> {
  std::vector<real_t> l_sizexi;
  for (std::size_t p{0}; p < m_extent.size(); p += 2) {
    l_sizexi.push_back(m_extent[p + 1] - m_extent[p]);
  }
  return l_sizexi;
}

auto Domain::dx() const -> real_t {
  assert(m_coord_system == CARTESIAN_COORD);
  return sizex1() / nx1();
}
auto Domain::dx1() const -> real_t { return sizex1() / nx1(); }
auto Domain::dx2() const -> real_t { return sizex2() / nx2(); }
auto Domain::dx3() const -> real_t { return sizex3() / nx3(); }
auto Domain::dxi() const -> std::vector<real_t> {
  std::vector<real_t> l_dxi;
  std::vector<real_t> l_sizexi{sizexi()};
  for (std::size_t p{0}; p < l_sizexi.size(); ++p) {
    l_dxi.push_back(l_sizexi[p] / static_cast<real_t>(m_resolution[p]));
  }
  return l_dxi;
}

auto Domain::dVol() const -> real_t {
  assert(m_dimension != UNDEFINED_D);
  if (m_coord_system == CARTESIAN_COORD) {
    if (m_dimension == ONE_D) {
      return dx();
    } else if (m_dimension == TWO_D) {
      return dx() * dx();
    } else {
      return dx() * dx() * dx();
    }
  }
  // TODO add different coord systems here
}

auto Domain::x1x2x3_to_ijk(std::vector<real_t> x1x2x3) -> std::vector<int> {
  std::vector<int> ijk;
  std::vector<real_t> l_dxi{dxi()};
  for (std::size_t p{0}; p < x1x2x3.size(); ++p) {
    ijk.push_back(static_cast<int>((x1x2x3[p] - m_extent[2 * p]) / l_dxi[p]));
  }
  return ijk;
}
auto Domain::ijk_to_x1x2x3(std::vector<int> ijk) -> std::vector<real_t> {
  std::vector<real_t> x1x2x3;
  std::vector<real_t> l_dxi{dxi()};
  for (std::size_t q{0}; q < ijk.size(); ++q) {
    x1x2x3.push_back(static_cast<real_t>(ijk[q]) * l_dxi[q] + m_extent[2 * q]);
  }
  return x1x2x3;
}

} // namespace ntt
