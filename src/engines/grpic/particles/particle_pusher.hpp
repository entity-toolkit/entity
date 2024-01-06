#ifndef GRPIC_PARTICLE_PUSHER_HPP
#define GRPIC_PARTICLE_PUSHER_HPP

#include "wrapper.h"

#include "sim_params.h"

#include "meshblock/meshblock.h"
#include "meshblock/particles.h"

#include "kernels/particle_pusher_gr.hpp"

namespace ntt {
  template <Dimension D, class M>
  void PushLoop(const SimulationParams&    params,
                Meshblock<D, GRPICEngine>& mblock,
                Particles<D, GRPICEngine>& species,
                real_t                     factor) {
    const real_t dt { factor * mblock.timestep() };
    const real_t charge_ovr_mass { species.mass() > ZERO
                                     ? species.charge() / species.mass()
                                     : ZERO };
    const real_t coeff { charge_ovr_mass * HALF * dt / params.larmor0() };

    auto kernel = Pusher_kernel<D, M>(mblock.em,
                                      mblock.em0,
                                      species.i1,
                                      species.i2,
                                      species.i3,
                                      species.i1_prev,
                                      species.i2_prev,
                                      species.i3_prev,
                                      species.dx1,
                                      species.dx2,
                                      species.dx3,
                                      species.dx1_prev,
                                      species.dx2_prev,
                                      species.dx3_prev,
                                      species.ux1,
                                      species.ux2,
                                      species.ux3,
                                      species.phi,
                                      species.tag,
                                      mblock.metric,
                                      coeff,
                                      dt,
                                      mblock.Ni1(),
                                      mblock.Ni2(),
                                      mblock.Ni3(),
                                      params.grPusherEpsilon(),
                                      params.grPusherNiter(),
                                      mblock.boundaries);

    if (species.pusher() == ParticlePusher::PHOTON) {
      // push photons
      auto range_policy = Kokkos::RangePolicy<AccelExeSpace, Photon_t>(
        0,
        species.npart());
      Kokkos::parallel_for("ParticlesPush", range_policy, kernel);
    } else if (species.pusher() == ParticlePusher::BORIS) {
      // push massive particles
      auto range_policy = Kokkos::RangePolicy<AccelExeSpace, Massive_t>(
        0,
        species.npart());
      Kokkos::parallel_for("ParticlesPush", range_policy, kernel);
    } else {
      NTTHostError("not implemented");
    }
  }
} // namespace ntt

#endif // GRPIC_PARTICLE_PUSHER_HPP
