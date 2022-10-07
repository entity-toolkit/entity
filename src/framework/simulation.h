#ifndef FRAMEWORK_SIMULATION_H
#define FRAMEWORK_SIMULATION_H

#include "wrapper.h"
#include "sim_params.h"
#include "meshblock.h"
#include "problem_generator.hpp"

#include <toml/toml.hpp>

#include <string>

namespace ntt {
  auto stringifySimulationType(const SimulationType&) -> std::string;
  auto stringifyBoundaryCondition(const BoundaryCondition&) -> std::string;
  auto stringifyParticlePusher(const ParticlePusher&) -> std::string;

  /**
   * @brief Main class of the simulation containing all the necessary methods and
   * configurations.
   * @tparam D dimension.
   * @tparam S simulation type.
   */
  template <Dimension D, SimulationType S>
  class Simulation {
  protected:
    // user-defined and inferred simulation parameters
    SimulationParams m_params;
    // time in physical units
    real_t m_time {0.0};
    // time in iteration timesteps
    std::size_t m_tstep {0};

  public:
    // problem setup generator
    ProblemGenerator<D, S> problem_generator;
    // meshblock with all the fields / metric / and particles
    Meshblock<D, S> meshblock;
    // random number pool
    RandomNumberPool_t random_pool;

    /**
     * @brief Constructor for simulation class.
     * @param inputdata toml-object with parsed toml parameters.
     */
    Simulation(const toml::value& inputdata);
    ~Simulation() = default;

    /**
     * @brief Initialize / allocate all the simulation objects based on the `m_params`.
     */
    void Initialize();

    /**
     * @brief Setup the problem using the problem generator.
     */
    void InitializeSetup();

    /**
     * @brief Verify that all the specified parameters are compatible before beginning the
     * simulation.
     */
    void Verify();

    /**
     * @brief Print all the simulation details using `plog`.
     */
    void PrintDetails();

    /**
     * @brief Finalize the simulation objects.
     */
    void Finalize();

    /**
     * @brief Get pointer to `m_params`.
     */
    [[nodiscard]] auto params() -> SimulationParams* { return &m_params; }

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
     * @brief Output the simulation data to a file.
     * @param tstep current timestep.
     */
    void WriteOutput(const unsigned long& tstep);

    /**
     * @brief Synchronize data from device to host.
     */
    void SynchronizeHostDevice();
  };

} // namespace ntt

#endif
