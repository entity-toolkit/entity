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
     */
    void faradaySubstep(const real_t& t, const real_t& f);

     /**
     * Compute E field.
     *
     * @param t time in physical units.
     */
    void Compute_E_Substep(const real_t& t);
     /**
     * Compute H field.
     *
     * @param t time in physical units.
     */
    void Compute_H_Substep(const real_t& t);
  };

} // namespace ntt

#endif