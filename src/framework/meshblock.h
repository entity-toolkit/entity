#ifndef FRAMEWORK_MESHBLOCK_H
#define FRAMEWORK_MESHBLOCK_H

#include "global.h"
#include "metric.h"
#include "fields.h"
#include "particles.h"

#include <vector>

namespace ntt {
  /**
   * Container for the meshgrid information (cell ranges etc).
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Mesh {
  protected:
    // active cell range in x1
    const int i_min, i_max;
    // active cell range in x2
    const int j_min, j_max;
    // active cell range in x3
    const int k_min, k_max;
    // number of active cells in each direction
    const std::size_t Ni, Nj, Nk;

  public:
    // Metric of the grid.
    std::shared_ptr<Metric<D>> metric;
    
    /**
     * Constructor for the mesh container, sets the active cell sizes and ranges.
     *
     * @param res resolution vector of size D (dimension).
     */
    Mesh(std::vector<std::size_t> res);
    ~Mesh() = default;

    /**
     * Loop over all active cells (disregard ghost cells).
     *
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto loopActiveCells() -> RangeND<D>;
    /**
     * Loop over all cells.
     *
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto loopAllCells() -> RangeND<D>;
  };

  /**
   * Container for the fields, particles and coordinate system. This is the main subject of the simulation.
   *
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

    /**
     * Constructor for the meshblock.
     *
     * @param res resolution vector of size D (dimension).
     * @param species vector of particle species parameters.
     */
    Meshblock(const std::vector<std::size_t>& res, const std::vector<ParticleSpecies>& species);
    ~Meshblock() = default;

    /**
     * Getters
     */
    [[nodiscard]] auto timestep() const -> const real_t& { return m_timestep; }
    [[nodiscard]] auto min_cell_size() const -> const real_t& { return m_min_cell_size; }

    /**
     * Setters
     */
    void set_timestep(const real_t& timestep) { m_timestep = timestep; }
    void set_min_cell_size(const real_t& min_cell_size) { m_min_cell_size = min_cell_size; }
  };

} // namespace ntt

#endif
