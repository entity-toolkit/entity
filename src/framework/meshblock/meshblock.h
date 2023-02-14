#ifndef FRAMEWORK_MESHBLOCK_H
#define FRAMEWORK_MESHBLOCK_H

#include "wrapper.h"

#include "fields.h"
#include "metric.h"
#include "particles.h"
#include "sim_params.h"

#include <vector>

namespace ntt {
  namespace dir {
    enum Direction { x = 1, y = 2, z = 3, r = 1, theta = 2, phi = 3, x1 = 1, x2 = 2, x3 = 3 };
  }    // namespace dir

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
    /** \See comments.hpp
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

    /* -------------------------------------------------------------------------- */
    /*                    Ranges in the device execution space                    */
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
    auto                           rangeActiveCells() -> range_t<D>;
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
    auto                           rangeAllCells() -> range_t<D>;

    /**
     * @brief Pick a particular region of cells.
     * @param boxRegion region of cells to pick: tuple of cellLayer objects.
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto                           rangeCells(const boxRegion<D>&) -> range_t<D>;
    /**
     * @brief Pick a particular region of cells.
     * @overload
     * @param range tuple of respective min and max ranges
     * @example {-1, 1} converts into {i_min - 1, i_max + 1} etc.
     * @example {{0, 0}, {0, 0}, {0, 0}} corresponds to allActiveLayer in all 3 dimensions.
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto               rangeCells(const tuple_t<tuple_t<int, Dim2>, D>&) -> range_t<D>;

    /* -------------------------------------------------------------------------- */
    /*                     Ranges in the host execution space                     */
    /* -------------------------------------------------------------------------- */
    /**
     * @brief Loop over all active cells (disregard ghost cells).
     * @returns Kokkos range policy in the host space with proper min/max indices and
     * dimension.
     */
    auto               rangeActiveCellsOnHost() -> range_h_t<D>;
    /**
     * @brief Loop over all cells.
     * @returns Kokkos range policy in the host space with proper min/max indices and
     * dimension.
     */
    auto               rangeAllCellsOnHost() -> range_h_t<D>;
    /**
     * @brief Pick a particular region of cells.
     * @param boxRegion region of cells to pick: tuple of cellLayer objects.
     * @returns Kokkos range policy in the host space with proper min/max indices and
     * dimension.
     */
    auto               rangeCellsOnHost(const boxRegion<D>&) -> range_h_t<D>;

    /**
     * @brief Get the first index of active zone along 1st dimension.
     */
    [[nodiscard]] auto i1_min() const -> const int& {
      return m_i1min;
    }
    /**
     * @brief Get the last index of active zone along 1st dimension.
     */
    [[nodiscard]] auto i1_max() const -> const int& {
      return m_i1max;
    }
    /**
     * @brief Get the number of active cells along 1st dimension.
     */
    [[nodiscard]] auto Ni1() const -> const int& {
      return m_Ni1;
    }
    /**
     * @brief Get the first index of active zone along 2nd dimension.
     */
    [[nodiscard]] auto i2_min() const -> const int& {
      return m_i2min;
    }
    /**
     * @brief Get the last index of active zone along 2nd dimension.
     */
    [[nodiscard]] auto i2_max() const -> const int& {
      return m_i2max;
    }
    /**
     * @brief Get the number of active cells along 2nd dimension.
     */
    [[nodiscard]] auto Ni2() const -> const int& {
      return m_Ni2;
    }
    /**
     * @brief Get the number of active cells along 3rd dimension.
     */
    [[nodiscard]] auto i3_min() const -> const int& {
      return m_i3min;
    }
    /**
     * @brief Get the last index of active zone along 3rd dimension.
     */
    [[nodiscard]] auto i3_max() const -> const int& {
      return m_i3max;
    }
    /**
     * @brief Get the number of active cells along 3rd dimension.
     */
    [[nodiscard]] auto Ni3() const -> const int& {
      return m_Ni3;
    }
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

    [[nodiscard]] auto extent() const -> std::vector<real_t>;
  };

  /**
   * @brief Container for the fields, particles and coordinate system. This is the main subject
   * of the simulation.
   * @tparam D Dimension.
   * @tparam S Simulation engine.
   */
  template <Dimension D, SimulationEngine S>
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
    [[nodiscard]] auto timestep() const -> const real_t& {
      return m_timestep;
    }
    /**
     * @brief Get the minimum cell size.
     */
    [[nodiscard]] auto minCellSize() const -> const real_t& {
      return m_min_cell_size;
    }

    /**
     * @brief Set the timestep of the meshblock.
     * @param timestep timestep in physical units.
     */
    void setTimestep(const real_t& timestep) {
      m_timestep = timestep;
    }
    /**
     * @brief Set the minimum cell size of the meshblock.
     * @param min_cell_size minimum cell size in physical units.
     */
    void setMinCellSize(const real_t& min_cell_size) {
      m_min_cell_size = min_cell_size;
    }

    /**
     * @brief Verify that all the specified parameters are valid.
     */
    void Verify();

    /**
     * @brief Remove dead particles.
     * @param max_dead_frac Maximum fraction of dead particles allowed ...
     * ... w.r.t. the living ones (npart).
     * @return Vector of the fraction of dead particles pre deletion.
     */
    auto RemoveDeadParticles(const double&) -> std::vector<double>;

    /* ----------------- Additional conversions and computations ---------------- */

    /**
     * @brief Fields to hatted basis.
     * Used for outputting/visualizing the fields.
     */
    void InterpolateAndConvertFieldsToHat();

    /**
     * @brief Currents to hatted basis.
     * Used for outputting/visualizing the currents.
     */
    void InterpolateAndConvertCurrentsToHat();

    /**
     * @brief Compute particle moment for output or other usage.
     * @param params SimulationParams object.
     * @param field FieldID for the moment to compute.
     * @param components Components of the field to compute (if applicable).
     * @param prtl_species Particle species to compute the moment for.
     * @param buff_ind Buffer index to store the result in (`meshblock::buff` array).
     * @param smooth Smoothing order (default: 2).
     *
     * @note Content of the meshblock::buff(*, buff_ind) has to be Content::empty.
     */
    void ComputeMoments(const SimulationParams& params,
                        const FieldID&          field,
                        const std::vector<int>& components,
                        const std::vector<int>& prtl_species,
                        const int&              buff_ind,
                        const short&            smooth = 2);

    /**
     * @brief Synchronize data from device to host.
     */
    void SynchronizeHostDevice(const SynchronizeFlags& flags = Synchronize_All);
  };

}    // namespace ntt

#endif
