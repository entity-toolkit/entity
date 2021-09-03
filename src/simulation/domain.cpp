#include "global.h"
#include "domain.h"

#include <plog/Log.h>

#include <cassert>
#include <vector>

namespace ntt {

void Domain::set_extent(std::vector<real_t> extent) {
  // check that everything is defined consistently
  if (m_dimension == ONE_D) {
    if (extent.size() > 2) {
      PLOGW << "1D simulation specified, ignoring extra dimensions in `extent`.";
      extent.erase(extent.begin() + 2, extent.end());
    }
  } else if (m_dimension == TWO_D) {
    if (extent.size() > 4) {
      PLOGW << "2D simulation specified, ignoring extra dimensions in `extent`.";
      extent.erase(extent.begin() + 4, extent.end());
    } else if (extent.size() < 4) {
      PLOGE << "2D simulation specified, not enough dimensions given in the input.";
      throw std::invalid_argument("Not enough values in `extent` input.");
    }
  } else if (m_dimension == THREE_D) {
    if (extent.size() > 6) {
      PLOGW << "3D simulation specified, ignoring extra dimensions in `extent`.";
      extent.erase(extent.begin() + 6, extent.end());
    } else if (extent.size() < 6) {
      PLOGE << "3D simulation specified, not enough dimensions given in the input.";
      throw std::invalid_argument("Not enough values in `extent` input.");
    }
  } else {
    throw std::runtime_error("# Error: unknown dimension of simulation.");
  }
  m_extent = extent;
}
void Domain::set_resolution(std::vector<int> resolution) {
  // check that everything is defined consistently
  if (m_dimension == ONE_D) {
    if (resolution.size() > 1) {
      PLOGW << "1D simulation specified, ignoring extra dimensions in `resolution`.";
      resolution.erase(resolution.begin() + 1, resolution.end());
    }
  } else if (m_dimension == TWO_D) {
    if (resolution.size() > 2) {
      PLOGW << "2D simulation specified, ignoring extra dimensions in `resolution`.";
      resolution.erase(resolution.begin() + 2, resolution.end());
    } else if (resolution.size() < 2) {
      PLOGE << "2D simulation specified, not enough dimensions given in the input.";
      throw std::invalid_argument("Not enough values in `resolution` input.");
    }
  } else if (m_dimension == THREE_D) {
    if (resolution.size() > 3) {
      PLOGW << "3D simulation specified, ignoring extra dimensions in `resolution`.";
      resolution.erase(resolution.begin() + 3, resolution.end());
    } else if (resolution.size() < 3) {
      PLOGE << "3D simulation specified, not enough dimensions given in the input.";
      throw std::invalid_argument("Not enough values in `resolution` input.");
    }
  } else {
    throw std::runtime_error("# Error: unknown dimension of simulation.");
  }
  m_resolution = resolution;
}

auto Domain::x1min() const -> real_t {
  assert((extent.size() >= 2) && (m_dimension != UNDEFINED_D));
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
  assert((m_resolution.size() == 3) && (m_dimension != UNDEFINED_D) && (m_dimension != ONE_D) && (m_dimension != TWO_D));
  return m_resolution[2];
}
auto Domain::sizex1() const -> real_t {
  return x1max() - x1min();
}
auto Domain::sizex2() const -> real_t {
  return x2max() - x2min();
}
auto Domain::sizex3() const -> real_t {
  return x3max() - x3min();
}

auto Domain::sizexi() const -> std::vector<real_t> {
  std::vector<real_t> sizeXi;
  for (std::size_t p {0}; p < m_extent.size(); p += 2) {
    sizeXi.push_back(m_extent[p + 1] - m_extent[p]);
  }
  return sizeXi;
}

auto Domain::dx1() const -> real_t {
  return sizex1() / nx1();
}
auto Domain::dx2() const -> real_t {
  return sizex2() / nx2();
}
auto Domain::dx3() const -> real_t {
  return sizex3() / nx3();
}
auto Domain::dxi() const -> std::vector<real_t> {
  std::vector<real_t> dXi;
  std::vector<real_t> sizeXi {sizexi()};
  for (std::size_t p {0}; p < sizeXi.size(); ++p) {
    dXi.push_back(sizeXi[p] / static_cast<real_t>(m_resolution[p]));
  }
  return dXi;
}

auto Domain::x1x2x3_to_ijk(std::vector<real_t> x1x2x3) -> std::vector<int> {
  std::vector<int> ijk;
  std::vector<real_t> dXi {dxi()};
  for (std::size_t p {0}; p < x1x2x3.size(); ++p) {
    ijk.push_back(static_cast<int>((x1x2x3[p] - m_extent[2 * p]) / dXi[p]));
  }
  return ijk;
}
auto Domain::ijk_to_x1x2x3(std::vector<int> ijk) -> std::vector<real_t> {
  std::vector<real_t> x1x2x3;
  std::vector<real_t> dXi {dxi()};
  for (std::size_t q {0}; q < ijk.size(); ++q) {
    x1x2x3.push_back(static_cast<real_t>(ijk[q]) * dXi[q]);
  }
  return x1x2x3;
}

}