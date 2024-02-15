/**
 * @file particle_pusher.cpp
 * @brief pushes the particle coordinates & velocities
 * @implements: `ParticlesPush` method of the `GRPIC` class
 * @includes: `kernels/particle_pusher_gr.hpp`
 * @depends: `grpic.h`
 *
 */

#include "wrapper.h"

#include "grpic.h"

#include "kernels/particle_pusher_gr.hpp"

#include METRIC_HEADER

namespace ntt {
  template <Dimension D>
  void GRPIC<D>::ParticlesPush(const real_t& factor) {
    auto&        mblock = this->meshblock;
    auto         params = *(this->params());
    const real_t dt { factor * mblock.timestep() };
    for (auto& species : mblock.particles) {
      if (species.npart() == 0 || species.pusher() == ParticlePusher::NONE) {
        continue;
      }
      const real_t charge_ovr_mass { species.mass() > ZERO
                                       ? species.charge() / species.mass()
                                       : ZERO };
      const real_t coeff { charge_ovr_mass * HALF * dt / params.larmor0() };
      auto         kernel { Pusher_kernel<D, Metric<D>>(mblock.em,
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
                                                mblock.boundaries) };
      if (species.pusher() == ParticlePusher::PHOTON) {
        // push photons
        auto range_policy = Kokkos::RangePolicy<AccelExeSpace, Massless_t>(
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
    NTTLog();
  }

} // namespace ntt

template void ntt::GRPIC<ntt::Dim2>::ParticlesPush(const real_t&);
template void ntt::GRPIC<ntt::Dim3>::ParticlesPush(const real_t&);