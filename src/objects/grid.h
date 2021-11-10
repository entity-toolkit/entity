#ifndef OBJECTS_GRID_H
#define OBJECTS_GRID_H

#include "global.h"

#include <vector>

namespace ntt {

template <Dimension D>
struct Grid {
  CoordinateSystem m_coord_system;
  std::vector<real_t> m_extent;
  std::vector<std::size_t> m_resolution;

  Grid(std::vector<std::size_t>);
  ~Grid() = default;

  void set_coord_system(const CoordinateSystem& coord_system) { m_coord_system = coord_system; }
  void set_extent(const std::vector<real_t>& extent) { m_extent = extent; }
  [[nodiscard]] auto get_dx1() const -> real_t {
    return (m_extent[1] - m_extent[0]) / static_cast<real_t>(m_resolution[0]);
  }
  [[nodiscard]] auto get_dx2() const -> real_t {
    if constexpr (D == ONE_D) {
      return 0.0;
    } else {
      return (m_extent[3] - m_extent[2]) / static_cast<real_t>(m_resolution[1]);
    }
  }
  [[nodiscard]] auto get_dx3() const -> real_t {
    if constexpr (D != THREE_D) {
      return 0.0;
    } else {
      return (m_extent[5] - m_extent[4]) / static_cast<real_t>(m_resolution[2]);
    }
  }
  [[nodiscard]] auto get_x1min() const -> real_t { return m_extent[0]; }
  [[nodiscard]] auto get_x1max() const -> real_t { return m_extent[1]; }
  [[nodiscard]] auto get_x2min() const -> real_t { return m_extent[2]; }
  [[nodiscard]] auto get_x2max() const -> real_t { return m_extent[3]; }
  [[nodiscard]] auto get_x3min() const -> real_t { return m_extent[4]; }
  [[nodiscard]] auto get_x3max() const -> real_t { return m_extent[5]; }
  [[nodiscard]] auto get_n1() const -> std::size_t { return m_resolution[0]; }
  [[nodiscard]] auto get_n2() const -> std::size_t { return m_resolution[1]; }
  [[nodiscard]] auto get_n3() const -> std::size_t { return m_resolution[2]; }

  Inline auto convert_iTOx1(const long int&) const -> real_t;
  Inline auto convert_jTOx2(const long int&) const -> real_t;
  Inline auto convert_kTOx3(const long int&) const -> real_t;

  // Accepts coordinate and returns the corresponding cell + ...
  // .. shift from the corner (normalized to cell size)
  Inline auto convert_x1TOidx1(const real_t&) const -> std::pair<long int, float>;
  Inline auto convert_x2TOjdx2(const real_t&) const -> std::pair<long int, float>;
  Inline auto convert_x3TOkdx3(const real_t&) const -> std::pair<long int, float>;

  // curvilinear-specific conversions
#ifndef HARDCODE_FLAT_COORDS
  Inline auto convert_x1TOx(const real_t&) const -> real_t;
  Inline auto convert_x1x2TOxy(const real_t&, const real_t&) const -> std::tuple<real_t, real_t>;
  Inline auto convert_x1x2x3TOxyz(const real_t&, const real_t&, const real_t&) const
      -> std::tuple<real_t, real_t, real_t>;

  Inline auto convert_xTOx1(const real_t&) const -> real_t;
  Inline auto convert_xyTOx1x2(const real_t&, const real_t&) const -> std::tuple<real_t, real_t>;
  Inline auto convert_xyzTOx1x2x3(const real_t&, const real_t&, const real_t&) const
      -> std::tuple<real_t, real_t, real_t>;

  // 1d
  Inline auto Jacobian_h1(const real_t&) const -> real_t;

  Inline auto Jacobian_11(const real_t&) const -> real_t;

  // 2d
  Inline auto Jacobian_h1(const real_t&, const real_t&) const -> real_t;
  Inline auto Jacobian_h2(const real_t&, const real_t&) const -> real_t;

  Inline auto Jacobian_11(const real_t&, const real_t&) const -> real_t;
  Inline auto Jacobian_12(const real_t&, const real_t&) const -> real_t;
  Inline auto Jacobian_21(const real_t&, const real_t&) const -> real_t;
  Inline auto Jacobian_22(const real_t&, const real_t&) const -> real_t;

  // 3d
  Inline auto Jacobian_h1(const real_t&, const real_t&, const real_t&) const -> real_t;
  Inline auto Jacobian_h2(const real_t&, const real_t&, const real_t&) const -> real_t;
  Inline auto Jacobian_h3(const real_t&, const real_t&, const real_t&) const -> real_t;

  Inline auto Jacobian_11(const real_t&, const real_t&, const real_t&) const -> real_t;
  Inline auto Jacobian_12(const real_t&, const real_t&, const real_t&) const -> real_t;
  Inline auto Jacobian_13(const real_t&, const real_t&, const real_t&) const -> real_t;
  Inline auto Jacobian_21(const real_t&, const real_t&, const real_t&) const -> real_t;
  Inline auto Jacobian_22(const real_t&, const real_t&, const real_t&) const -> real_t;
  Inline auto Jacobian_23(const real_t&, const real_t&, const real_t&) const -> real_t;
  Inline auto Jacobian_31(const real_t&, const real_t&, const real_t&) const -> real_t;
  Inline auto Jacobian_32(const real_t&, const real_t&, const real_t&) const -> real_t;
  Inline auto Jacobian_33(const real_t&, const real_t&, const real_t&) const -> real_t;
#endif
};

template <Dimension D>
Inline auto Grid<D>::convert_iTOx1(const long int& i) const -> real_t {
  return m_extent[0]
         + (static_cast<real_t>(i - N_GHOSTS) / static_cast<real_t>(m_resolution[0]))
               * (m_extent[1] - m_extent[0]);
}
template <Dimension D>
Inline auto Grid<D>::convert_jTOx2(const long int& j) const -> real_t {
  return m_extent[2]
         + (static_cast<real_t>(j - N_GHOSTS) / static_cast<real_t>(m_resolution[1]))
               * (m_extent[3] - m_extent[2]);
}
template <Dimension D>
Inline auto Grid<D>::convert_kTOx3(const long int& k) const -> real_t {
  return m_extent[4]
         + (static_cast<real_t>(k - N_GHOSTS) / static_cast<real_t>(m_resolution[2]))
               * (m_extent[5] - m_extent[4]);
}

// Accepts coordinate and returns the corresponding cell + ...
// .. shift from the corner (normalized to cell size)
template <Dimension D>
Inline auto Grid<D>::convert_x1TOidx1(const real_t& x1) const -> std::pair<long int, float> {
  // TESTPERF: floor vs something else
  real_t dx1 {
      (x1 - m_extent[0]) / ((m_extent[1] - m_extent[0]) / static_cast<real_t>(m_resolution[0]))};
  long int i {static_cast<long int>(std::floor(dx1))};
  dx1 = dx1 - static_cast<real_t>(i);
  return {i + N_GHOSTS, dx1};
}
template <Dimension D>
Inline auto Grid<D>::convert_x2TOjdx2(const real_t& x2) const -> std::pair<long int, float> {
  real_t dx2 {
      (x2 - m_extent[2]) / ((m_extent[3] - m_extent[2]) / static_cast<real_t>(m_resolution[1]))};
  long int j {static_cast<long int>(std::floor(dx2))};
  dx2 = dx2 - static_cast<real_t>(j);
  return {j + N_GHOSTS, dx2};
}
template <Dimension D>
Inline auto Grid<D>::convert_x3TOkdx3(const real_t& x3) const -> std::pair<long int, float> {
  real_t dx3 {
      (x3 - m_extent[4]) / ((m_extent[5] - m_extent[4]) / static_cast<real_t>(m_resolution[2]))};
  long int k {static_cast<long int>(std::floor(dx3))};
  dx3 = dx3 - static_cast<real_t>(k);
  return {k + N_GHOSTS, dx3};
}

// curvilinear-specific conversions
#ifndef HARDCODE_FLAT_COORDS

// curvilinear-to-cartesian
template <>
Inline auto Grid<ONE_D>::convert_x1TOx(const real_t& x1) const -> real_t {
#  ifdef HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  else
  throw std::runtime_error("# Error: dimensionality and coord system incompatible.");
#  endif
}

template <>
Inline auto Grid<TWO_D>::convert_x1x2TOxy(const real_t& x1, const real_t& x2) const
    -> std::tuple<real_t, real_t> {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return {x1 * std::cos(x2), x1 * std::sin(x2)};
#  elif HARDCODE_CYLINDRICAL_COORDS
  return {x1 * std::cos(x2), x1 * std::sin(x2)};
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto
Grid<THREE_D>::convert_x1x2x3TOxyz(const real_t& x1, const real_t& x2, const real_t& x3) const
    -> std::tuple<real_t, real_t, real_t> {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return {x1 * std::sin(x2) * std::cos(x3), x1 * std::sin(x2) * std::sin(x3), x1 * std::cos(x2)};
#  elif HARDCODE_CYLINDRICAL_COORDS
  return {x1 * std::cos(x2), x1 * std::sin(x2), x3};
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

// cartesian-to-curvilinear
template <>
Inline auto Grid<ONE_D>::convert_xTOx1(const real_t& x) const -> real_t {
#  ifdef HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  else
  throw std::runtime_error("# Error: dimensionality and coord system incompatible.");
#  endif
}

template <>
Inline auto Grid<TWO_D>::convert_xyTOx1x2(const real_t& x, const real_t& y) const
    -> std::tuple<real_t, real_t> {
#  ifdef HARDCODE_SPHERICAL_COORDS
  real_t r {std::sqrt(x * x + y * y)};
  return {r, std::acos(y / r)};
#  elif HARDCODE_CYLINDRICAL_COORDS
  real_t r {std::sqrt(x * x + y * y)};
  return {r, std::atan(y / x) + PI * HEAVISIDE(-x)};
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto
Grid<THREE_D>::convert_xyzTOx1x2x3(const real_t& x, const real_t& y, const real_t& z) const
    -> std::tuple<real_t, real_t, real_t> {
#  ifdef HARDCODE_SPHERICAL_COORDS
  real_t r {std::sqrt(x * x + y * y + z * z)};
  return {r, std::acos(z / r), std::atan(y / x) + PI * HEAVISIDE(-x)};
#  elif HARDCODE_CYLINDRICAL_COORDS
  real_t r {std::sqrt(x * x + y * y)};
  return {r, std::atan(y / x) + PI * HEAVISIDE(-x), z};
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

// Jacobian coefficients
// h-values
// 1d
template <>
Inline auto Grid<ONE_D>::Jacobian_h1(const real_t& x1) const -> real_t {
#  ifdef HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  else
  throw std::runtime_error("# Error: dimensionality and coord system incompatible.");
#  endif
}

// 2d
template <>
Inline auto Grid<TWO_D>::Jacobian_h1(const real_t& x1, const real_t& x2) const -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return ONE;
#  elif HARDCODE_CYLINDRICAL_COORDS
  return ONE;
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto Grid<TWO_D>::Jacobian_h2(const real_t& x1, const real_t& x2) const -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return x1;
#  elif HARDCODE_CYLINDRICAL_COORDS
  return x1;
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

// 3d
template <>
Inline auto Grid<THREE_D>::Jacobian_h1(const real_t& x1, const real_t& x2, const real_t& x3) const
    -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return ONE;
#  elif HARDCODE_CYLINDRICAL_COORDS
  return ONE;
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto Grid<THREE_D>::Jacobian_h2(const real_t& x1, const real_t& x2, const real_t& x3) const
    -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return x1;
#  elif HARDCODE_CYLINDRICAL_COORDS
  return x1;
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto Grid<THREE_D>::Jacobian_h3(const real_t& x1, const real_t& x2, const real_t& x3) const
    -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return x1 * std::sin(x2);
#  elif HARDCODE_CYLINDRICAL_COORDS
  return ONE;
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

// Matrix
// 1d
template <>
Inline auto Grid<ONE_D>::Jacobian_11(const real_t& x1) const -> real_t {
#  ifdef HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  else
  throw std::runtime_error("# Error: dimensionality and coord system incompatible.");
#  endif
}

// 2d
template <>
Inline auto Grid<TWO_D>::Jacobian_11(const real_t& x1, const real_t& x2) const -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return std::sin(x2);
#  elif HARDCODE_CYLINDRICAL_COORDS
  return std::cos(x2);
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto Grid<TWO_D>::Jacobian_12(const real_t& x1, const real_t& x2) const -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return x1 * std::cos(x2);
#  elif HARDCODE_CYLINDRICAL_COORDS
  return -x1 * std::sin(x2);
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto Grid<TWO_D>::Jacobian_21(const real_t& x1, const real_t& x2) const -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return std::sin(x2);
#  elif HARDCODE_CYLINDRICAL_COORDS
  return std::sin(x2);
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto Grid<TWO_D>::Jacobian_22(const real_t& x1, const real_t& x2) const -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return x1 * std::cos(x2);
#  elif HARDCODE_CYLINDRICAL_COORDS
  return x1 * std::cos(x2);
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

// 3d
template <>
Inline auto Grid<THREE_D>::Jacobian_11(const real_t& x1, const real_t& x2, const real_t& x3) const
    -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return std::sin(x2) * std::cos(x3);
#  elif HARDCODE_CYLINDRICAL_COORDS
  return std::cos(x2);
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto Grid<THREE_D>::Jacobian_12(const real_t& x1, const real_t& x2, const real_t& x3) const
    -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return x1 * std::cos(x2) * std::cos(x3);
#  elif HARDCODE_CYLINDRICAL_COORDS
  return -x1 * std::sin(x2);
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto Grid<THREE_D>::Jacobian_13(const real_t& x1, const real_t& x2, const real_t& x3) const
    -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return -x1 * std::sin(x2) * std::sin(x3);
#  elif HARDCODE_CYLINDRICAL_COORDS
  return ZERO;
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto Grid<THREE_D>::Jacobian_21(const real_t& x1, const real_t& x2, const real_t& x3) const
    -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return std::sin(x2) * std::sin(x3);
#  elif HARDCODE_CYLINDRICAL_COORDS
  return std::sin(x2);
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto Grid<THREE_D>::Jacobian_22(const real_t& x1, const real_t& x2, const real_t& x3) const
    -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return x1 * std::cos(x2) * std::sin(x3);
#  elif HARDCODE_CYLINDRICAL_COORDS
  return x1 * std::cos(x2);
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto Grid<THREE_D>::Jacobian_23(const real_t& x1, const real_t& x2, const real_t& x3) const
    -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return x1 * std::sin(x2) * std::cos(x3);
#  elif HARDCODE_CYLINDRICAL_COORDS
  return ZERO;
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto Grid<THREE_D>::Jacobian_31(const real_t& x1, const real_t& x2, const real_t& x3) const
    -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return std::cos(x2);
#  elif HARDCODE_CYLINDRICAL_COORDS
  return ZERO;
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto Grid<THREE_D>::Jacobian_32(const real_t& x1, const real_t& x2, const real_t& x3) const
    -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return -x1 * std::sin(x2);
#  elif HARDCODE_CYLINDRICAL_COORDS
  return ZERO;
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

template <>
Inline auto Grid<THREE_D>::Jacobian_33(const real_t& x1, const real_t& x2, const real_t& x3) const
    -> real_t {
#  ifdef HARDCODE_SPHERICAL_COORDS
  return ZERO;
#  elif HARDCODE_CYLINDRICAL_COORDS
  return ONE;
#  elif HARDCODE_CARTESIAN_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_SPHERICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  throw std::logic_error("# NOT IMPLEMENTED.");
#  endif
}

#endif

} // namespace ntt

#endif
