#ifndef FRAMEWORK_PGEN_H
#define FRAMEWORK_PGEN_H

#include "global.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

  /**
   * Parent class for all the problem generators (virtual).
   *
   * @tparam D dimension.
   * @tparam S simulation type.
   */
  template <Dimension D, SimulationType S>
  struct PGen {
    PGen(const SimulationParams&) {}
    ~PGen() = default;
    /**
     * Field initializer for the problem generator.
     *
     * @param sim_params simulation parameters object.
     * @param mblock meshblock object.
     */
    virtual void userInitFields(const SimulationParams&, Meshblock<D, S>&) {}
    /**
     * Particle initializer for the problem generator.
     *
     * @param sim_params simulation parameters object.
     * @param mblock meshblock object.
     */
    virtual void userInitParticles(const SimulationParams&, Meshblock<D, S>&) {}
    /**
     * Field boundary conditions for the problem generator.
     *
     * @param sim_params simulation parameters object.
     * @param mblock meshblock object.
     */
    virtual void userBCFields(const real_t&, const SimulationParams&, Meshblock<D, S>&) {}

    /**
     * Target radial B-field for absorbing boundaries.
     *
     * @param mblock meshblock object.
     * @param x coordinate in code units (array of size D).
     * @return radial component of the target B-field in hatted basis.
     */
    virtual Inline auto userTargetField_br_hat(const Meshblock<D, S>&, const coord_t<D>&) const -> real_t { return ZERO; }
  };

} // namespace ntt

#endif
