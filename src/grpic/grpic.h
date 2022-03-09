#ifndef GRPIC_GRPIC_H
#define GRPIC_GRPIC_H

#include "global.h"
#include "simulation.h"

#include <toml/toml.hpp>

namespace ntt {
  /**
   * Class for GRPIC simulations, inherits from `Simulation<D, SimulationType::GRPIC>`.
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  class GRPIC : public Simulation<D, SimulationType::GRPIC> {
  public:
    /**
     * Constructor for GRPIC class.
     *
     * @param inputdata toml-object with parsed toml parameters.
     */
    GRPIC(const toml::value& inputdata) : Simulation<D, SimulationType::GRPIC>(inputdata) {}
    ~GRPIC() = default;

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
     * From the initial fields, advances the first time steps.
     * 
     * @param t time in physical units
     */
    void initial_step(const real_t&);

    /**
     * Advance the simulation forward for a specified amount of timesteps, keeping track of time.
     */
    void mainloop();

    /**
     * Process the simulation (calling initialize, verify, mainloop, etc).
     */
    void process();

     /**
     * Advance B-field using Faraday's law.
     *
     * @param t time in physical units.
     * @param f coefficient that gets multiplied by the timestep (e.g., 0.5).
     * @param s switches whether it applies on em0 or em
     */
    void faradaySubstep(const real_t& t, const real_t& f, const short& s);
 
    /**
     * Advance D-field using Ampere's law.
     *
     * @param t time in physical units.
     * @param f coefficient that gets multiplied by the timestep (e.g., 0.5).
     * @param s switches whether it applies to em0 or em
     */
    void ampereSubstep(const real_t& t, const real_t& f, const short& s);
 
     /**
     * Compute E field.
     *
     * @param t time in physical units.
     * @param s switches whether it applies to em0 or em
     */
    void Compute_E_Substep(const real_t& t, const short& s);
  
     /**
     * Compute H field.
     *
     * @param t time in physical units.
     * @param s switches whether it applies to em0 or em
     */
    void Compute_H_Substep(const real_t& t, const short& s);
  
    /**
     * Time average EM fields.
     *
     * @param t time in physical units.
     * @param s switches whether it applies to em0 or em
     */
    void Average_EM_Substep(const real_t& t);
  
     /**
     * Time average currents.
     *
     * @param t time in physical units.
     */
    void Average_J_Substep(const real_t& t);
 
     /**
     * Apply boundary conditions for fields.
     *
     * @param t time in physical units.
     * @param s switches whether it applies to B or D
     */
    void fieldBoundaryConditions(const real_t& t, const short& s);

     /**
     * Apply boundary conditions for auxiliary fields.
     *
     * @param t time in physical units.
     */
    void AuxiliaryBoundaryConditions(const real_t& t);
  };

} // namespace ntt

#endif