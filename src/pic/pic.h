#ifndef PIC_PIC_H
#define PIC_PIC_H

#include "global.h"
#include "simulation.h"

#include <toml/toml.hpp>

namespace ntt {
  /**
   * @brief Class for PIC simulations, inherits from `Simulation<D, SimulationType::PIC>`.
   * @tparam D dimension.
   */
  template <Dimension D>
  class PIC : public Simulation<D, SimulationType::PIC> {
  public:
    /**
     * @brief Constructor for PIC class.
     * @param inputdata toml-object with parsed toml parameters.
     */
    PIC(const toml::value& inputdata) : Simulation<D, SimulationType::PIC>(inputdata) {}
    ~PIC() = default;

    /**
     * @brief Advance the simulation forward for one timestep.
     * @param t time in physical units
     */
    void step_forward(const real_t& t);

    /**
     * @brief Advance the simulation forward for one timestep.
     * @param t time in physical units
     */
    void step_backward(const real_t& t);

    /**
     * @brief Advance the simulation forward for a specified amount of timesteps, keeping track
     * of time.
     */
    void mainloop();

    /**
     * @brief Process the simulation (calling initialize, verify, mainloop, etc).
     */
    void process();

    /**
     * @brief Dummy function to match with GRPIC
     * @param time in physical units
     */
    void initial_step(const real_t&) {}
    /**
     * @brief Reset field arrays.
     * @param t time in physical units.
     */
    void resetFields(const real_t& t);
    /**
     * @brief Reset current arrays.
     * @param t time in physical units.
     */
    void resetCurrents(const real_t& t);
    /**
     * @brief Reset particles.
     * @param t time in physical units.
     */
    void resetParticles(const real_t& t);

    /**
     * @brief Advance B-field using Faraday's law.
     * @param t time in physical units.
     * @param f coefficient that gets multiplied by the timestep (e.g., 0.5).
     */
    void faradaySubstep(const real_t& t, const real_t& f);
    /**
     * @brief Advance E-field using Ampere's law (without currents).
     * @param t time in physical units.
     * @param f coefficient that gets multiplied by the timestep (e.g., 1.0).
     */
    void ampereSubstep(const real_t& t, const real_t& f);
    /**
     * @brief Deposit currents from particles.
     * @param t time in physical units.
     */
    void depositCurrentsSubstep(const real_t& t);
    /**
     * @brief Add computed and filtered currents to the E-field.
     * @param t time in physical units.
     */
    void addCurrentsSubstep(const real_t& t);
    /**
     * @brief Spatially filter all the deposited currents.
     * @param t time in physical units.
     */
    void filterCurrentsSubstep(const real_t& t);

    /**
     * @brief Advance particle positions and velocities.
     * @param t time in physical units.
     * @param f coefficient that gets multiplied by the timestep (e.g., 1.0).
     */
    void pushParticlesSubstep(const real_t& t, const real_t& f);
    /**
     * @brief Apply boundary conditions for fields.
     * @param t time in physical units.
     */
    void fieldBoundaryConditions(const real_t& t);
    /**
     * @brief Apply boundary conditions for currents.
     * @param t time in physical units.
     */
    void currentBoundaryConditions(const real_t& t);
    /**
     * @brief Apply boundary conditions for particles.
     * @param t time in physical units.
     */
    void particleBoundaryConditions(const real_t& t);

    /**
     * @brief Benchmarking step.
     */
    void benchmark();
  };

} // namespace ntt

#endif
