#ifndef FRAMEWORK_MESH_H
#define FRAMEWORK_MESH_H

#include "wrapper.h"

#include METRIC_HEADER

#include <vector>

namespace ntt {
  namespace dir {
    enum Direction {
      x     = 1,
      y     = 2,
      z     = 3,
      r     = 1,
      theta = 2,
      phi   = 3,
      x1    = 1,
      x2    = 2,
      x3    = 3
    };
  } // namespace dir

  enum class CellLayer {
    allLayer,
    activeLayer,
    minGhostLayer,
    minActiveLayer,
    maxActiveLayer,
    maxGhostLayer
  };
  /**
   * min/max layers have N_GHOSTS cells
   *
   * allLayer:                 .* *|* * * * * * * * *\* *.
   * activeLayer:              .   |* * * * * * * * *\   .
   * minGhostLayer:            .* *|                 \   .
   * minActiveLayer:           .   |* *              \   .
   * maxActiveLayer:           .   |              * *\   .
   * maxGhostLayer:            .   |                 \* *.
   *
   */

  /**
   * @brief Usage example:
   * 1. boxRegion<Dim2>{ CellLayer::minGhostLayer, CellLayer::maxGhostLayer }
   * results in a region [ [ i1min - N_GHOSTS, i1min ),
   *                       [ i2max, i2max + N_GHOSTS ) ]
   *
   * 2. boxRegion<Dim3>{ CellLayer::activeLayer,
   *                     CellLayer::maxActiveLayer,
   *                     CellLayer::allLayer }
   * results in a region [ [ i1min, i1max ),
   *                       [ i2max - N_GHOSTS, i2max ),
   *                       [ i3min - N_GHOSTS, i3max + N_GHOSTS ) ]
   */
  template <Dimension D>
  using boxRegion = tuple_t<CellLayer, D>;

  /**
   * @brief Container for the meshgrid information (cell ranges etc).
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Mesh {
  protected:
    /**
     *                                          N_GHOSTS
     *                                            __
     *       |<--            Ni1              -->|  |
     *    ...........................................
     *    .                                         .
     *    .                                         .  <-- i2_max
     *    .  ^===================================^  .___________________
     *    .  |                                   \  .                   |
     *    .  |                                   \  .                   |
     *    .  |                                   \  .                   |
     *    .  |                                   \  .                   |-- Ni2
     *    .  |                                   \  .                   |
     *    .  |                                   \  .                   |
     *    .  |                                   \  .  <-- i2_min       |
     *    .  ^-----------------------------------^  . __________________|
     *    .                                         .   |
     *    .                                         .   |- N_GHOSTS
     *    ........................................... __|
     *        ^                                   ^
     *        |                                   |
     *      i1_min                             i1_max
     */

    // active cell range in x1
    const unsigned int m_i1min, m_i1max;
    // active cell range in x2
    const unsigned int m_i2min, m_i2max;
    // active cell range in x3
    const unsigned int m_i3min, m_i3max;
    // number of active cells in each direction
    const unsigned int m_Ni1, m_Ni2, m_Ni3;

  public:
    // Metric of the grid.
    Metric<D> metric;

    /**
     * @brief Constructor for the mesh container, sets the active cell sizes and ranges.
     * @param res resolution vector of size D (dimension).
     * @param ext extent vector of size 2 * D.
     * @param params metric-/domain-specific parameters (max: 10).
     */
    Mesh(const std::vector<unsigned int>& res,
         const std::vector<real_t>&       ext,
         const real_t*                    params);
    ~Mesh() = default;

    // Boundary conditions.
    std::vector<std::vector<BoundaryCondition>> boundaries;

    /* -------------------------------------------------------------------------- */
    /*                    Ranges in the device execution space */
    /* -------------------------------------------------------------------------- */
    /**
     * @brief Loop over all active cells (disregard ghost cells).
     * @returns Kokkos range policy with proper min/max indices and dimension.
     *
     *    . . . . . . . . . . . . .
     *    .                       .
     *    .                       .
     *    .   ^= = = = = = = =^   .
     *    .   |* * * * * * * *\   .
     *    .   |* * * * * * * *\   .
     *    .   |* * * * * * * *\   .
     *    .   |* * * * * * * *\   .
     *    .   ^- - - - - - - -^   .
     *    .                       .
     *    .                       .
     *    . . . . . . . . . . . . .
     */
    auto rangeActiveCells() -> range_t<D>;
    /**
     * @brief Loop over all cells.
     * @returns Kokkos range policy with proper min/max indices and dimension.
     *
     *    . . . . . . . . . . . . .
     *    .* * * * * * * * * * * *.
     *    .* * * * * * * * * * * *.
     *    .* *^= = = = = = = =^* *.
     *    .* *|* * * * * * * *\* *.
     *    .* *|* * * * * * * *\* *.
     *    .* *|* * * * * * * *\* *.
     *    .* *|* * * * * * * *\* *.
     *    .* *^- - - - - - - -^* *.
     *    .* * * * * * * * * * * *.
     *    .* * * * * * * * * * * *.
     *    . . . . . . . . . . . . .
     *
     */
    auto rangeAllCells() -> range_t<D>;

    /**
     * @brief Pick a particular region of cells.
     * @param boxRegion region of cells to pick: tuple of cellLayer objects.
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto rangeCells(const boxRegion<D>&) -> range_t<D>;
    /**
     * @brief Pick a particular region of cells.
     * @overload
     * @param range tuple of respective min and max ranges
     * @example {-1, 1} converts into {i_min - 1, i_max + 1} etc.
     * @example {{0, 0}, {0, 0}, {0, 0}} corresponds to allActiveLayer in all 3 dimensions.
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto rangeCells(const tuple_t<tuple_t<int, Dim2>, D>&) -> range_t<D>;

    /* -------------------------------------------------------------------------- */
    /*                     Ranges in the host execution space */
    /* -------------------------------------------------------------------------- */
    /**
     * @brief Loop over all active cells (disregard ghost cells).
     * @returns Kokkos range policy in the host space with proper min/max
     * indices and dimension.
     */
    auto rangeActiveCellsOnHost() -> range_h_t<D>;
    /**
     * @brief Loop over all cells.
     * @returns Kokkos range policy in the host space with proper min/max
     * indices and dimension.
     */
    auto rangeAllCellsOnHost() -> range_h_t<D>;
    /**
     * @brief Pick a particular region of cells.
     * @param boxRegion region of cells to pick: tuple of cellLayer objects.
     * @returns Kokkos range policy in the host space with proper min/max
     * indices and dimension.
     */
    auto rangeCellsOnHost(const boxRegion<D>&) -> range_h_t<D>;

    /**
     * @brief Get the first index of active zone along 1st dimension.
     */
    [[nodiscard]]
    auto i1_min() const -> unsigned int {
      return m_i1min;
    }

    /**
     * @brief Get the last index of active zone along 1st dimension.
     */
    [[nodiscard]]
    auto i1_max() const -> unsigned int {
      return m_i1max;
    }

    /**
     * @brief Get the number of active cells along 1st dimension.
     */
    [[nodiscard]]
    auto Ni1() const -> unsigned int {
      return m_Ni1;
    }

    /**
     * @brief Get the first index of active zone along 2nd dimension.
     */
    [[nodiscard]]
    auto i2_min() const -> unsigned int {
      return m_i2min;
    }

    /**
     * @brief Get the last index of active zone along 2nd dimension.
     */
    [[nodiscard]]
    auto i2_max() const -> unsigned int {
      return m_i2max;
    }

    /**
     * @brief Get the number of active cells along 2nd dimension.
     */
    [[nodiscard]]
    auto Ni2() const -> unsigned int {
      return m_Ni2;
    }

    /**
     * @brief Get the number of active cells along 3rd dimension.
     */
    [[nodiscard]]
    auto i3_min() const -> unsigned int {
      return m_i3min;
    }

    /**
     * @brief Get the last index of active zone along 3rd dimension.
     */
    [[nodiscard]]
    auto i3_max() const -> unsigned int {
      return m_i3max;
    }

    /**
     * @brief Get the number of active cells along 3rd dimension.
     */
    [[nodiscard]]
    auto Ni3() const -> unsigned int {
      return m_Ni3;
    }

    /**
     * @brief Get the first index of active zone along i-th dimension.
     */
    [[nodiscard]]
    auto i_min(short i) const -> unsigned int {
      switch (i) {
        case 0:
          return m_i1min;
        case 1:
          return m_i2min;
        case 2:
          return m_i3min;
        default:
          NTTHostError("Invalid dimension");
      }
    }

    /**
     * @brief Get the last index of active zone along i-th dimension.
     */
    [[nodiscard]]
    auto i_max(short i) const -> unsigned int {
      switch (i) {
        case 0:
          return m_i1max;
        case 1:
          return m_i2max;
        case 2:
          return m_i3max;
        default:
          NTTHostError("Invalid dimension");
      }
    }

    /**
     * @brief Get the number of active cells along i-th dimension.
     */
    [[nodiscard]]
    auto Ni(short i) const -> unsigned int {
      switch (i) {
        case 0:
          return m_Ni1;
        case 1:
          return m_Ni2;
        case 2:
          return m_Ni3;
        default:
          NTTHostError("Invalid dimension");
      }
    }

    [[nodiscard]]
    auto extent() const -> std::vector<real_t>;
  };
} // namespace ntt

#endif