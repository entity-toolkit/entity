#ifndef PIC_PIC_H
#define PIC_PIC_H

#include "global.h"
#include "simulation.h"

#include <toml/toml.hpp>

namespace ntt {
  /**
   * Class for PIC simulations, inherits from `Simulation<D, SimulationType::PIC>`.
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  class PIC : public Simulation<D, SimulationType::PIC> {
  public:
    /**
     * Constructor for PIC class.
     *
     * @param inputdata toml-object with parsed toml parameters.
     */
    PIC(const toml::value& inputdata) : Simulation<D, SimulationType::PIC>(inputdata) {}
    ~PIC() = default;

    /**
     * Advance the simulation forward for one timestep.
     *
     * @param t time in physical units
     */
    void step_forward(const real_t&);

    /**
     * Advance the simulation forward for one timestep.
     *
     * @param t time in physical units
     */
    void step_backward(const real_t&);

    /**
     * Advance the simulation forward for a specified amount of timesteps, keeping track of time.
     */
    void mainloop();

    /**
     * Process the simulation (calling initialize, verify, mainloop, etc).
     */
    void process();

    /**
     * Dummy function to match with GRPIC
     *
     */
    void initial_step(const real_t&) {}

    /**
     * Advance B-field using Faraday's law.
     *
     * @param t time in physical units.
     * @param f coefficient that gets multiplied by the timestep (e.g., 0.5).
     */
    void faradaySubstep(const real_t&, const real_t&);
    /**
     * Advance E-field using Ampere's law (without currents).
     *
     * @param t time in physical units.
     * @param f coefficient that gets multiplied by the timestep (e.g., 1.0).
     */
    void ampereSubstep(const real_t&, const real_t&);
    /**
     * Add computed and filtered currents to the E-field.
     *
     * @param t time in physical units.
     */
    void addCurrentsSubstep(const real_t&);
    /**
     * Reset current arrays.
     *
     * @param t time in physical units.
     */
    void resetCurrentsSubstep(const real_t&);

    /**
     * Advance particle positions and velocities.
     *
     * @param t time in physical units.
     * @param f coefficient that gets multiplied by the timestep (e.g., 1.0).
     */
    void pushParticlesSubstep(const real_t&, const real_t&);
    /**
     * Deposit currents from particles.
     *
     * @param t time in physical units.
     */
    void depositCurrentsSubstep(const real_t&);

    /**
     * Apply boundary conditions for fields.
     *
     * @param t time in physical units.
     */
    void fieldBoundaryConditions(const real_t&);
    /**
     * Apply boundary conditions for particles.
     *
     * @param t time in physical units.
     */
    void particleBoundaryConditions(const real_t&);
  };

} // namespace ntt

#endif