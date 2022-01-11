#ifndef FRAMEWORK_PGEN_H
#define FRAMEWORK_PGEN_H

#include "global.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct PGen {
    PGen(const SimulationParams&) {}
    ~PGen() = default;
    virtual void userInitFields(const SimulationParams&, Meshblock<D, S>&) {}
    virtual void userInitParticles(const SimulationParams&, Meshblock<D, S>&) {}
    virtual void userBCFields(const real_t&, const SimulationParams&, Meshblock<D, S>&) {}

    // Inline auto userTargetField_br_HAT(Meshblock<D>&, const real_t&) const -> real_t { return ZERO; }
    // Inline auto userTargetField_br_HAT(Meshblock<D>&, const real_t&, const real_t&) const -> real_t { return ZERO; }
    // Inline auto userTargetField_br_HAT(Meshblock<D>&, const real_t&, const real_t&, const real_t&) const -> real_t {
    //   return ZERO;
    // }
  };

} // namespace ntt

#endif
