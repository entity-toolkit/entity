#ifndef OBJECTS_GRID_H
#define OBJECTS_GRID_H

#include "global.h"
#include "coord_system.h"

#include <vector>
#include <memory>

namespace ntt {

  template <Dimension D>
  struct Grid {
    NTTArray<real_t*> grid_x1;
    NTTArray<real_t*> grid_x2;
    NTTArray<real_t*> grid_x3;

    std::shared_ptr<CoordinateSystem<D>> m_coord_system;
    const std::vector<real_t> m_extent;
    const std::vector<std::size_t> m_resolution;

    Grid(std::vector<real_t>, std::vector<std::size_t>);
    ~Grid() = default;

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

    // this implies uniform grid in (x1, x2, x3)
    Inline auto convert_iTOx1(const long int&) const -> real_t;
    Inline auto convert_jTOx2(const long int&) const -> real_t;
    Inline auto convert_kTOx3(const long int&) const -> real_t;

    // Accepts coordinate and returns the corresponding cell + ...
    // .. shift from the corner (normalized to cell size)
    Inline auto convert_x1TOidx1(const real_t&) const -> std::pair<long int, float>;
    Inline auto convert_x2TOjdx2(const real_t&) const -> std::pair<long int, float>;
    Inline auto convert_x3TOkdx3(const real_t&) const -> std::pair<long int, float>;

    [[nodiscard]] auto get_imin() const -> long int { return N_GHOSTS; }
    [[nodiscard]] auto get_imax() const -> long int { return N_GHOSTS + m_resolution[0]; }
    [[nodiscard]] auto get_jmin() const -> long int { return N_GHOSTS; }
    [[nodiscard]] auto get_jmax() const -> long int { return N_GHOSTS + m_resolution[1]; }
    [[nodiscard]] auto get_kmin() const -> long int { return N_GHOSTS; }
    [[nodiscard]] auto get_kmax() const -> long int { return N_GHOSTS + m_resolution[2]; }

    auto loopActiveCells() -> RangeND<D>;
    auto loopAllCells() -> RangeND<D>;

    auto loopCells(const long int&, const long int&) -> RangeND<D>;
    auto loopCells(const long int&, const long int&, const long int&, const long int&) -> RangeND<D>;
    auto loopCells(const long int&, const long int&, const long int&, const long int&, const long int&, const long int&) -> RangeND<D>;
  };

  template <Dimension D>
  Inline auto Grid<D>::convert_iTOx1(const long int& i) const -> real_t {
    return m_extent[0] + (
              static_cast<real_t>(i - N_GHOSTS) / static_cast<real_t>(m_resolution[0])
              ) * (m_extent[1] - m_extent[0]);
  }
  template <Dimension D>
  Inline auto Grid<D>::convert_jTOx2(const long int& j) const -> real_t {
    return m_extent[2] + (
              static_cast<real_t>(j - N_GHOSTS) / static_cast<real_t>(m_resolution[1])
              ) * (m_extent[3] - m_extent[2]);
  }
  template <Dimension D>
  Inline auto Grid<D>::convert_kTOx3(const long int& k) const -> real_t {
    return m_extent[4] + (
              static_cast<real_t>(k - N_GHOSTS) / static_cast<real_t>(m_resolution[2])
              ) * (m_extent[5] - m_extent[4]);
  }

  // Accepts coordinate and returns the corresponding cell + ...
  // .. shift from the corner (normalized to cell size)
  template <Dimension D>
  Inline auto Grid<D>::convert_x1TOidx1(const real_t& x1) const -> std::pair<long int, float> {
    // TESTPERF: floor vs something else
    real_t dx1 {(x1 - m_extent[0]) / ((m_extent[1] - m_extent[0]) / static_cast<real_t>(m_resolution[0]))};
    long int i {static_cast<long int>(std::floor(dx1))};
    dx1 = dx1 - static_cast<real_t>(i);
    return {i + N_GHOSTS, dx1};
  }
  template <Dimension D>
  Inline auto Grid<D>::convert_x2TOjdx2(const real_t& x2) const -> std::pair<long int, float> {
    real_t dx2 {(x2 - m_extent[2]) / ((m_extent[3] - m_extent[2]) / static_cast<real_t>(m_resolution[1]))};
    long int j {static_cast<long int>(std::floor(dx2))};
    dx2 = dx2 - static_cast<real_t>(j);
    return {j + N_GHOSTS, dx2};
  }
  template <Dimension D>
  Inline auto Grid<D>::convert_x3TOkdx3(const real_t& x3) const -> std::pair<long int, float> {
    real_t dx3 {(x3 - m_extent[4]) / ((m_extent[5] - m_extent[4]) / static_cast<real_t>(m_resolution[2]))};
    long int k {static_cast<long int>(std::floor(dx3))};
    dx3 = dx3 - static_cast<real_t>(k);
    return {k + N_GHOSTS, dx3};
  }

} // namespace ntt

#endif
