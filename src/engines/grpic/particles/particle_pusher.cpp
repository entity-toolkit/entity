#include "particle_pusher.hpp"

#include "wrapper.h"

#include "grpic.h"

namespace ntt {
  template <Dimension D>
  void GRPIC<D>::ParticlesPush(const real_t& factor) {
    auto& mblock = this->meshblock;
    auto  params = *(this->params());
    for (auto& species : mblock.particles) {
      const real_t dt { factor * mblock.timestep() };
      const real_t charge_ovr_mass { species.mass() > ZERO
                                       ? species.charge() / species.mass()
                                       : ZERO };
      const real_t coeff { charge_ovr_mass * HALF * dt / params.larmor0() };

      if (species.pusher() == ParticlePusher::PHOTON) {
        // push photons
        auto range_policy = Kokkos::RangePolicy<AccelExeSpace, Photon_t>(
          0,
          species.npart());
        Kokkos::parallel_for("ParticlesPush",
                             range_policy,
                             Pusher_kernel<D>(mblock,
                                              species,
                                              coeff,
                                              dt,
                                              params.grPusherEpsilon(),
                                              params.grPusherNiter()));
      } else if (species.pusher() == ParticlePusher::BORIS) {
        // push massive particles
        auto range_policy = Kokkos::RangePolicy<AccelExeSpace, Massive_t>(
          0,
          species.npart());
        Kokkos::parallel_for("ParticlesPush",
                             range_policy,
                             Pusher_kernel<D>(mblock,
                                              species,
                                              coeff,
                                              dt,
                                              params.grPusherEpsilon(),
                                              params.grPusherNiter()));
      } else if (species.pusher() == ParticlePusher::NONE) {
        // do nothing
      } else {
        NTTHostError("not implemented");
      }
    }
    NTTLog();
  }

} // namespace ntt

template void ntt::GRPIC<ntt::Dim2>::ParticlesPush(const real_t&);
template void ntt::GRPIC<ntt::Dim3>::ParticlesPush(const real_t&);