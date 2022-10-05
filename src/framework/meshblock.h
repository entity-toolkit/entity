#ifndef FRAMEWORK_MESHBLOCK_H
#define FRAMEWORK_MESHBLOCK_H

#include "global.h"
#include "metric.h"
#include "fields.h"
#include "particles.h"

#include <vector>

namespace ntt {
  enum class CellLayer {
    allLayer,
    activeLayer,
    minGhostLayer,
    minActiveLayer,
    maxActiveLayer,
    maxGhostLayer
  };
  /**
   * "layer":           N_GHOSTS cells
   *
   * minGhostLayer                       maxActiveLayer
   *    |                                   |
   *    |  minActiveLayer                   |  maxGhostLayer
   *    |     |                             |     |
   * [--+-----+---------- allLayer ---------+-----+--]
   *    |     |                             |     |
   *    |  [--+--------- activeLayer -------+--]  |
   *    |     |                             |     |
   *    |     |                             |     |
   * |  v  |  v  |                       |  v  |  v  |
   * X=====O=====X=======================X=====O=====X
   *       |                                   |
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
    // active cell range in x1
    const int m_i1min, m_i1max;
    // active cell range in x2
    const int m_i2min, m_i2max;
    // active cell range in x3
    const int m_i3min, m_i3max;
    // number of active cells in each direction
    const int m_Ni1, m_Ni2, m_Ni3;

  public:
    // Metric of the grid.
    const Metric<D> metric;

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
    std::vector<BoundaryCondition> boundaries;

    /**
     * @brief Loop over all active cells (disregard ghost cells).
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto rangeActiveCells() -> range_t<D>;
    /**
     * @brief Loop over all cells.
     * @returns Kokkos range policy with proper min/max indices and dimension.
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
    auto rangeCells(const tuple_t<tuple_t<short, Dim2>, D>&) -> range_t<D>;

    /**
     * @brief Get the first index of active zone along 1st dimension.
     */
    [[nodiscard]] auto i1_min() const -> const int& { return m_i1min; }
    /**
     * @brief Get the last index of active zone along 1st dimension.
     */
    [[nodiscard]] auto i1_max() const -> const int& { return m_i1max; }
    /**
     * @brief Get the number of active cells along 1st dimension.
     */
    [[nodiscard]] auto Ni1() const -> const int& { return m_Ni1; }
    /**
     * @brief Get the first index of active zone along 2nd dimension.
     */
    [[nodiscard]] auto i2_min() const -> const int& { return m_i2min; }
    /**
     * @brief Get the last index of active zone along 2nd dimension.
     */
    [[nodiscard]] auto i2_max() const -> const int& { return m_i2max; }
    /**
     * @brief Get the number of active cells along 2nd dimension.
     */
    [[nodiscard]] auto Ni2() const -> const int& { return m_Ni2; }
    /**
     * @brief Get the number of active cells along 3rd dimension.
     */
    [[nodiscard]] auto i3_min() const -> const int& { return m_i3min; }
    /**
     * @brief Get the last index of active zone along 3rd dimension.
     */
    [[nodiscard]] auto i3_max() const -> const int& { return m_i3max; }
    /**
     * @brief Get the number of active cells along 3rd dimension.
     */
    [[nodiscard]] auto Ni3() const -> const int& { return m_Ni3; }
    /**
     * @brief Get the first index of active zone along i-th dimension.
     */
    [[nodiscard]] auto i_min(short i) const -> const int& {
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
    [[nodiscard]] auto i_max(short i) const -> const int& {
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
    [[nodiscard]] auto Ni(short i) const -> const int& {
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
  };

  /**
   * @brief Container for the fields, particles and coordinate system. This is the main subject
   * of the simulation.
   * @tparam D Dimension.
   * @tparam S Simulation type.
   */
  template <Dimension D, SimulationType S>
  class Meshblock : public Mesh<D>, public Fields<D, S> {
  private:
    // Timestep duration in physical units defined at the meshblock.
    real_t m_timestep;
    // Effective minimum cell size of the meshblock.
    real_t m_min_cell_size;

  public:
    // Vector of particles species.
    std::vector<Particles<D, S>> particles;
    RandomNumberPool_t*          random_pool_ptr;

    /**
     * @brief Constructor for the meshblock.
     * @param res resolution vector of size D (dimension).
     * @param ext extent vector of size 2 * D.
     * @param params metric-/domain-specific parameters (max: 10).
     * @param species vector of particle species parameters.
     */
    Meshblock(const std::vector<unsigned int>&    res,
              const std::vector<real_t>&          ext,
              const real_t*                       params,
              const std::vector<ParticleSpecies>& species);
    ~Meshblock() = default;

    /**
     * @brief Get the timestep.
     */
    [[nodiscard]] auto timestep() const -> const real_t& { return m_timestep; }
    /**
     * @brief Get the minimum cell size.
     */
    [[nodiscard]] auto min_cell_size() const -> const real_t& { return m_min_cell_size; }

    /**
     * @brief Set the timestep of the meshblock.
     * @param timestep timestep in physical units.
     */
    void set_timestep(const real_t& timestep) { m_timestep = timestep; }
    /**
     * @brief Set the minimum cell size of the meshblock.
     * @param min_cell_size minimum cell size in physical units.
     */
    void set_min_cell_size(const real_t& min_cell_size) { m_min_cell_size = min_cell_size; }
  };

} // namespace ntt

#endif
