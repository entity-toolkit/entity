#ifndef FRAMEWORK_SIMULATION_H
#define FRAMEWORK_SIMULATION_H

#include "global.h"
#include "sim_params.h"
#include "meshblock.h"
#include "problem_generator.hpp"

#include <toml/toml.hpp>

#include <string>

namespace ntt {
  /**
   * @brief Main class of the simulation containing all the necessary methods and configurations.
   * @tparam D dimension.
   * @tparam S simulation type.
   */
  template <Dimension D, SimulationType S>
  class Simulation {
  protected:
    // user-defined and inferred simulation parameters
    SimulationParams m_sim_params;
    // problem setup generator
    ProblemGenerator<D, S> m_pGen;
    // meshblock with all the fields / metric / and particles
    Meshblock<D, S> m_mblock;

  public:
    /**
     * @brief Constructor for simulation class.
     * @param inputdata toml-object with parsed toml parameters.
     */
    Simulation(const toml::value& inputdata);
    ~Simulation() = default;

    /**
     * @brief Initialize / allocate all the simulation objects based on the `m_sim_params`.
     */
    void initialize();

    /**
     * @brief Setup the problem using the problem generator.
     */
    void initializeSetup();

    /**
     * @brief Verify that all the specified parameters are compatible before beginning the simulation.
     */
    void verify();

    /**
     * @brief Print all the simulation details using `plog`.
     */
    void printDetails();

    /**
     * @brief Finalize the simulation objects.
     */
    void finalize();

    /**
     * @brief Get pointer to `sim_params`.
     */
    [[nodiscard]] auto sim_params() -> SimulationParams* { return &m_sim_params; }
    /**
     * @brief Get pointer to `mblock`.
     */
    [[nodiscard]] auto mblock() -> Meshblock<D, S>* { return &m_mblock; }
    /**
     * @brief Get pointer to `pgen`.
     */
    [[nodiscard]] auto pgen() -> ProblemGenerator<D, S>* { return &m_pGen; }

    /**
     * @brief Loop over all active cells (disregard ghost cells).
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto rangeActiveCells() -> RangeND<D>;
    /**
     * @brief Loop over all cells.
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto rangeAllCells() -> RangeND<D>;
  };

} // namespace ntt

#endif
