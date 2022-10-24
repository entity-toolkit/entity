#ifndef FRAMEWORK_PGEN_H
#define FRAMEWORK_PGEN_H

#include "wrapper.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct PGen {
    virtual inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) {}
    virtual inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) {}
    virtual inline void
    UserBCFields(const real_t&, const SimulationParams&, Meshblock<D, S>&) {}
    // Inline virtual auto UserTargetField_br_hat(const Meshblock<D, S>&, const coord_t<D>&) const
    //   -> real_t {
    //   return ZERO;
    // }
    virtual inline void
    UserDriveParticles(const real_t&, const SimulationParams&, Meshblock<D, S>&) {}
  };

} // namespace ntt

template struct ntt::PGen<ntt::Dim1, ntt::TypePIC>;
template struct ntt::PGen<ntt::Dim2, ntt::TypePIC>;
template struct ntt::PGen<ntt::Dim3, ntt::TypePIC>;

#endif // FRAMEWORK_PGEN_H