#ifndef FRAMEWORK_SIMULATION_H
#define FRAMEWORK_SIMULATION_H

#include "wrapper.h"

#include "sim_params.h"

#include "io/output.h"
#include "io/writer.h"
#include "meshblock/meshblock.h"
#include "utils/timer.h"

#include <toml.hpp>

#include <string>

namespace ntt {
  auto stringizeSimulationEngine(const SimulationEngine&) -> std::string;
  auto stringizeBoundaryCondition(const BoundaryCondition&) -> std::string;
  auto stringizeParticlePusher(const ParticlePusher&) -> std::string;

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
    // meshblock with all the fields / metric / and particles
    Meshblock<D, S>    meshblock;
    // writer
    Writer<D, S>       writer;
    // random number pool
    RandomNumberPool_t random_pool;

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
     * @brief Diagnostic logging.
     * @param os output stream.
     */
    void               PrintDiagnostics(const std::size_t&         step,
                                        const real_t&              time,
                                        const std::vector<double>& fractions,
                                        const timer::Timers&       timer,
                                        std::vector<long double>&  tstep_durations,
                                        std::ostream&              os = std::cout);

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
    auto rangeActiveCells() -> range_t<D>;

    /**
     * @brief Loop over all cells.
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto rangeAllCells() -> range_t<D>;
  };

}    // namespace ntt

#endif
