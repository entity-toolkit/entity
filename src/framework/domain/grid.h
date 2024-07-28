/**
 * @file framework/domain/grid.h
 * @brief Grid class containing information about the discretization
 * @implements
 *   - ntt::Grid<>
 * @cpp:
 *   - grid.cpp
 * @namespaces:
 *   - ntt::
 * @note
 * Schematics for the grid in 2D:
 *
 *                          N_GHOSTS
 *                            __
 *       |<--      Ni1    -->|  |
 *    ...........................
 *    .                         .
 *    .                         .  <-- i2_max
 *    .  ^===================^  .___________________
 *    .  |                   \  .                   |
 *    .  |                   \  .                   |
 *    .  |                   \  .                   |-- Ni2
 *    .  |                   \  .                   |
 *    .  |                   \  .  <-- i2_min       |
 *    .  ^-------------------^  . __________________|
 *    .                         .   |
 *    .                         .   |- N_GHOSTS
 *    ........................... __|
 *        ^                   ^
 *        |                   |
 *      i1_min             i1_max
 * @note
 * Grid::RangeActiveCells
 *    . . . . . . . . . . .
 *    .                   .
 *    .                   .
 *    .   ^= = = = = =^   .
 *    .   |* * * * * *\   .
 *    .   |* * * * * *\   .
 *    .   |* * * * * *\   .
 *    .   |* * * * * *\   .
 *    .   ^- - - - - -^   .
 *    .                   .
 *    .                   .
 *    . . . . . . . . . . .
 *
 * Grid::RangeAllCells
 *    . . . . . . . . . .
 *    .* * * * * * * * *.
 *    .* * * * * * * * *.
 *    .* *^= = = = =^* *.
 *    .* *|* * * * *\* *.
 *    .* *|* * * * *\* *.
 *    .* *|* * * * *\* *.
 *    .* *^- - - - -^* *.
 *    .* * * * * * * * *.
 *    .* * * * * * * * *.
 *    . . . . . . . . . .
 *
 */

#ifndef FRAMEWORK_DOMAIN_GRID_H
#define FRAMEWORK_DOMAIN_GRID_H

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

#include <vector>

namespace ntt {

  template <Dimension D>
  struct Grid {
    Grid(const std::vector<std::size_t>& res) : m_resolution { res } {
      raise::ErrorIf(m_resolution.size() != D, "invalid dimension", HERE);
    }

    ~Grid() = default;

    /* getters -------------------------------------------------------------- */
    [[nodiscard]]
    auto i_min(in i) const -> std::size_t {
      switch (i) {
        case in::x1:
          return (m_resolution.size() > 0) ? N_GHOSTS : 0;
        case in::x2:
          return (m_resolution.size() > 1) ? N_GHOSTS : 0;
        case in::x3:
          return (m_resolution.size() > 2) ? N_GHOSTS : 0;
        default:
          raise::Error("invalid dimension", HERE);
          throw;
      }
    }

    [[nodiscard]]
    auto i_max(in i) const -> std::size_t {
      switch (i) {
        case in::x1:
          return (m_resolution.size() > 0) ? (m_resolution[0] + N_GHOSTS) : 1;
        case in::x2:
          return (m_resolution.size() > 1) ? (m_resolution[1] + N_GHOSTS) : 1;
        case in::x3:
          return (m_resolution.size() > 2) ? (m_resolution[2] + N_GHOSTS) : 1;
        default:
          raise::Error("invalid dimension", HERE);
          throw;
      }
    }

    [[nodiscard]]
    auto n_active(in i) const -> std::size_t {
      switch (i) {
        case in::x1:
          return (m_resolution.size() > 0) ? m_resolution[0] : 1;
        case in::x2:
          return (m_resolution.size() > 1) ? m_resolution[1] : 1;
        case in::x3:
          return (m_resolution.size() > 2) ? m_resolution[2] : 1;
        default:
          raise::Error("invalid dimension", HERE);
          throw;
      }
    }

    [[nodiscard]]
    auto n_active() const -> std::vector<std::size_t> {
      return m_resolution;
    }

    [[nodiscard]]
    auto n_all(in i) const -> std::size_t {
      switch (i) {
        case in::x1:
          return (m_resolution.size() > 0) ? (m_resolution[0] + 2 * N_GHOSTS) : 1;
        case in::x2:
          return (m_resolution.size() > 1) ? (m_resolution[1] + 2 * N_GHOSTS) : 1;
        case in::x3:
          return (m_resolution.size() > 2) ? (m_resolution[2] + 2 * N_GHOSTS) : 1;
        default:
          raise::Error("invalid dimension", HERE);
          throw;
      }
    }

    [[nodiscard]]
    auto n_all() const -> std::vector<std::size_t> {
      std::vector<std::size_t> nall;
      for (std::size_t i = 0; i < D; ++i) {
        nall.push_back(m_resolution[i] + 2 * N_GHOSTS);
      }
      return nall;
    }

    /* Ranges in the device execution space --------------------------------- */
    /**
     * @brief Loop over all active cells (disregard ghost cells)
     * @returns Kokkos range policy with proper min/max indices and dimension
     */
    auto rangeActiveCells() const -> range_t<D>;
    /**
     * @brief Loop over all cells
     * @returns Kokkos range policy with proper min/max indices and dimension
     */
    auto rangeAllCells() const -> range_t<D>;

    /**
     * @brief Pick a particular region of cells
     * @param box_region_t region of cells to pick: tuple of cellLayer objects
     * @returns Kokkos range policy with proper min/max indices and dimension
     */
    auto rangeCells(const box_region_t<D>&) const -> range_t<D>;
    /**
     * @brief Pick a particular region of cells
     * @overload
     * @param range tuple of respective min and max ranges
     * @example {-1, 1} converts into {i_min - 1, i_max + 1} etc
     * @example {{0, 0}, {0, 0}, {0, 0}} corresponds to allActiveLayer in all 3 dimensions
     * @returns Kokkos range policy with proper min/max indices and dimension
     */
    auto rangeCells(const tuple_t<list_t<int, 2>, D>&) const -> range_t<D>;

    /* Ranges in the host execution space ----------------------------------- */
    /**
     * @brief Loop over all active cells (disregard ghost cells)
     * @returns Kokkos range policy in the host space with proper min/max
     * indices and dimension.
     */
    auto rangeActiveCellsOnHost() const -> range_h_t<D>;
    /**
     * @brief Loop over all cells
     * @returns Kokkos range policy in the host space with proper min/max
     * indices and dimension.
     */
    auto rangeAllCellsOnHost() const -> range_h_t<D>;
    /**
     * @brief Pick a particular region of cells
     * @param box_region_t region of cells to pick: tuple of cellLayer objects
     * @returns Kokkos range policy in the host space with proper min/max
     * indices and dimension.
     */
    auto rangeCellsOnHost(const box_region_t<D>&) const -> range_h_t<D>;

  protected:
    std::vector<std::size_t> m_resolution;
  };

} // namespace ntt

#endif // FRAMEWORK_DOMAIN_GRID_H
