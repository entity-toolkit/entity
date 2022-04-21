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
   * Main class of the simulation containing all the necessary methods and configurations.
   *
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
     * Constructor for simulation class.
     *
     * @param inputdata toml-object with parsed toml parameters.
     */
    Simulation(const toml::value& inputdata);
    ~Simulation() = default;

    /**
     * Initialize / allocate all the simulation objects based on the `m_sim_params`.
     */
    void initialize();

    /**
     * Setup the problem using the problem generator.
     */
    void initializeSetup();

    /**
     * Verify that all the specified parameters are compatible before beginning the simulation.
     */
    void verify();

    /**
     * Print all the simulation details using `plog`.
     */
    void printDetails();

    /**
     * Finalize the simulation objects.
     */
    void finalize();

    /**
     * Getters.
     */
    [[nodiscard]] auto sim_params() const -> const SimulationParams& { return m_sim_params; }
    [[nodiscard]] auto mblock() const -> const Meshblock<D, S>& { return m_mblock; }
    [[nodiscard]] auto pgen() const -> const ProblemGenerator<D, S>& { return m_pGen; }
  };

} // namespace ntt

#endif
