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
  };

} // namespace ntt

#endif