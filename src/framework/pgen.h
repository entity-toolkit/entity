#ifndef FRAMEWORK_PGEN_H
#define FRAMEWORK_PGEN_H

#include "wrapper.h"
#include "sim_params.h"
#include "meshblock.h"

#ifdef NTTINY_ENABLED
#  include "nttiny/api.h"
#endif

#include <map>

namespace ntt {

  template <Dimension D, SimulationType S>
  struct PGen {
    virtual inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) {}
    virtual inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) {}
    virtual inline void
    UserBCFields(const real_t&, const SimulationParams&, Meshblock<D, S>&) {}
    virtual inline void
    UserDriveParticles(const real_t&, const SimulationParams&, Meshblock<D, S>&) {}

#ifdef NTTINY_ENABLED
    virtual inline void
    UserInitBuffers_nttiny(const SimulationParams&,
                           const Meshblock<D, S>&,
                           std::map<std::string, nttiny::ScrollingBuffer>&) {}
    virtual inline void
    UserSetBuffers_nttiny(const real_t&,
                          const SimulationParams&,
                          const Meshblock<D, S>&,
                          std::map<std::string, nttiny::ScrollingBuffer>&) {}
#endif
  };

} // namespace ntt

#endif // FRAMEWORK_PGEN_H