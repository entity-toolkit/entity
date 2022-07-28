#ifndef FRAMEWORK_MESHBLOCK_H
#define FRAMEWORK_MESHBLOCK_H

#include "global.h"
#include "metric.h"
#include "fields.h"
#include "particles.h"

#include <vector>

namespace ntt {
  /**
   * @brief Container for the meshgrid information (cell ranges etc).
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Mesh {
  protected:
    // active cell range in x1
    const int m_imin, m_imax;
    // active cell range in x2
    const int m_jmin, m_jmax;
    // active cell range in x3
    const int m_kmin, m_kmax;
    // number of active cells in each direction
    const int m_Ni, m_Nj, m_Nk;

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

    /**
     * @brief Loop over all active cells (disregard ghost cells).
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto loopActiveCells() -> RangeND<D>;
    /**
     * @brief Loop over all cells.
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto loopAllCells() -> RangeND<D>;

    /**
     * @brief Get the first index of active zone along 1st dimension.
     */
    [[nodiscard]] auto i_min() const -> const int& { return m_imin; }
    /**
     * @brief Get the last index of active zone along 1st dimension.
     */
    [[nodiscard]] auto i_max() const -> const int& { return m_imax; }
    /**
     * @brief Get the number of active cells along 1st dimension.
     */
    [[nodiscard]] auto Ni() const -> const int& { return m_Ni; }
    /**
     * @brief Get the first index of active zone along 2nd dimension.
     */
    [[nodiscard]] auto j_min() const -> const int& { return m_jmin; }
    /**
     * @brief Get the last index of active zone along 2nd dimension.
     */
    [[nodiscard]] auto j_max() const -> const int& { return m_jmax; }
    /**
     * @brief Get the number of active cells along 2nd dimension.
     */
    [[nodiscard]] auto Nj() const -> const int& { return m_Nj; }
    /**
     * @brief Get the number of active cells along 3rd dimension.
     */
    [[nodiscard]] auto k_min() const -> const int& { return m_kmin; }
    /**
     * @brief Get the last index of active zone along 3rd dimension.
     */
    [[nodiscard]] auto k_max() const -> const int& { return m_kmax; }
    /**
     * @brief Get the number of active cells along 3rd dimension.
     */
    [[nodiscard]] auto Nk() const -> const int& { return m_Nk; }
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
    // Boundary conditions.
    std::vector<BoundaryCondition> boundaries;

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
