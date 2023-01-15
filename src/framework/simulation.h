#ifndef FRAMEWORK_SIMULATION_H
#define FRAMEWORK_SIMULATION_H

#include "wrapper.h"

#include "fields.h"
#include "meshblock.h"
#include "sim_params.h"
#include "writer.h"

#include "problem_generator.hpp"

#include <toml/toml.hpp>

#include <string>

namespace ntt {
  auto stringifySimulationEngine(const SimulationEngine&) -> std::string;
  auto stringifyBoundaryCondition(const BoundaryCondition&) -> std::string;
  auto stringifyParticlePusher(const ParticlePusher&) -> std::string;

  /**
   * @brief Main class of the simulation containing all the necessary methods and
   * configurations.
   * @tparam D dimension.
   * @tparam S simulation engine.
   */
  template <Dimension D, SimulationEngine S>
  class Simulation {
  protected:
    // user-defined and inferred simulation parameters
    SimulationParams m_params;
    // time in physical units
    real_t           m_time { 0.0 };
    // time in iteration timesteps
    std::size_t      m_tstep { 0 };

  public:
    // problem setup generator
    ProblemGenerator<D, S> problem_generator;
    // meshblock with all the fields / metric / and particles
    Meshblock<D, S>        meshblock;
    // writer
    Writer<D, S>           writer;
    // random number pool
    RandomNumberPool_t     random_pool;

    /**
     * @brief Constructor for simulation class.
     * @param inputdata toml-object with parsed toml parameters.
     */
    Simulation(const toml::value& inputdata);
    ~Simulation() = default;

    /* -------------------------------------------------------------------------- */
    /*                                Main routines                               */
    /* -------------------------------------------------------------------------- */
    /**
     * @brief Initialize / allocate all the simulation objects based on the `m_params`.
     */
    void               Initialize();

    /**
     * @brief Setup the problem using the problem generator.
     */
    void               InitializeSetup();

    /**
     * @brief Verify that all the specified parameters are compatible before beginning the
     * simulation.
     */
    void               Verify();

    /**
     * @brief Print all the simulation details using `plog`.
     */
    void               PrintDetails();

    /**
     * @brief Finalize the simulation objects.
     */
    void               Finalize();

    /**
     * @brief Synchronize data from device to host.
     */
    void               SynchronizeHostDevice();

    /**
     * @brief Diagnostic logging.
     * @param os output stream.
     */
    void               PrintDiagnostics(std::ostream& os = std::cout);

    /* -------------------------------------------------------------------------- */
    /*                                   Getters                                  */
    /* -------------------------------------------------------------------------- */
    /**
     * @brief Get pointer to `m_params`.
     */
    [[nodiscard]] auto params() -> SimulationParams* {
      return &m_params;
    }
    /**
     * @brief Get the physical time
     */
    [[nodiscard]] auto time() const -> real_t {
      return m_time;
    }
    /**
     * @brief Get the current timestep
     */
    [[nodiscard]] auto tstep() const -> std::size_t {
      return m_tstep;
    }

    /**
     * @brief Loop over all active cells (disregard ghost cells).
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto         rangeActiveCells() -> range_t<D>;

    /**
     * @brief Loop over all cells.
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto         rangeAllCells() -> range_t<D>;

    /* -------------------------------------------------------------------------- */
    /*                                 Converters                                 */
    /* -------------------------------------------------------------------------- */
    /**
     * @brief Fields on the host to hatted basis.
     * Used for outputting/visualizing the fields.
     */
    virtual void InterpolateAndConvertFieldsToHat() = 0;
  };

}    // namespace ntt

#endif
