#ifndef OBJECTS_SIMULATION_H
#define OBJECTS_SIMULATION_H

#include "global.h"
#include "sim_params.h"
#include "meshblock.h"
#include "pgen.h"

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
    SimulationParams m_sim_params;
    PGen<D, S> m_pGen;
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
     * Initialize / allocate all the simulation objects based on the `m_sim_params`
     *
     * @param inputdata toml-object with parsed toml parameters.
     */
    void initialize();
    // void userInitialize();
    void verify();
    void printDetails();
    // void finalize();

    // void step_forward(const real_t&);
    // void step_backward(const real_t&);
    // void mainloop();

    void run();

    // // fields
    // void faradaySubstep(const real_t&, const real_t&);
    // void ampereSubstep(const real_t&, const real_t&);
    // void addCurrentsSubstep(const real_t&);
    // void resetCurrentsSubstep(const real_t&);
    // // particles
    // void pushParticlesSubstep(const real_t&);
    // void depositSubstep(const real_t&);
    // // boundaries
    // void fieldBoundaryConditions(const real_t&);
    // void particleBoundaryConditions(const real_t&);

    // [[nodiscard]] auto get_params() const -> const SimulationParams& { return m_sim_params; }
    // [[nodiscard]] auto get_meshblock() const -> const Meshblock<D>& { return mblock; }
  };

} // namespace ntt

#endif
