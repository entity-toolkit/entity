#include "particle_pusher.hpp"

#include "wrapper.h"

#include "pic.h"

namespace ntt {
  template <Dimension D>
  void PIC<D>::ParticlesPush(const real_t& factor) {
    auto& mblock = this->meshblock;
    auto  params = *(this->params());
    auto  time   = this->m_time;
    for (auto& species : mblock.particles) {
      const real_t dt { factor * mblock.timestep() };
      const real_t charge_ovr_mass { species.mass() > ZERO ? species.charge() / species.mass()
                                                           : ZERO };
      const real_t coeff { charge_ovr_mass * HALF * dt / params.larmor0() };

      if (species.pusher() == ParticlePusher::PHOTON) {
        // push photons
        auto range_policy = Kokkos::RangePolicy<AccelExeSpace, Photon_t>(0, species.npart());
#ifdef EXTERNAL_FORCE
        auto&            pgen = this->problem_generator;
        array_t<real_t*> work { "work", species.npart() };
        Kokkos::parallel_for("ParticlesPush",
                             range_policy,
                             Pusher_kernel<D>(mblock, species, pgen, work, time, coeff, dt));
#else
        Kokkos::parallel_for(
          "ParticlesPush", range_policy, Pusher_kernel<D>(mblock, species, time, coeff, dt));
#endif
      } else if (species.pusher() == ParticlePusher::BORIS) {
        // push boris-particles
        auto range_policy = Kokkos::RangePolicy<AccelExeSpace, Boris_t>(0, species.npart());
#ifdef EXTERNAL_FORCE
        auto&            pgen = this->problem_generator;
        array_t<real_t*> work { "work", species.npart() };
        Kokkos::parallel_for("ParticlesPush",
                             range_policy,
                             Pusher_kernel<D>(mblock, species, pgen, work, time, coeff, dt));
        real_t global_sum = 0.0;
        Kokkos::parallel_reduce(
          "ParticlesPush",
          species.npart(),
          Lambda(index_t p, real_t & sum) { sum += work(p); },
          global_sum);
        problem_generator.work_done += global_sum;
#else
        Kokkos::parallel_for(
          "ParticlesPush", range_policy, Pusher_kernel<D>(mblock, species, time, coeff, dt));
#endif
      } else if (species.pusher() == ParticlePusher::NONE) {
        // do nothing
      } else {
        NTTHostError("not implemented");
      }
    }
    NTTLog();
  }

}    // namespace ntt

template void ntt::PIC<ntt::Dim1>::ParticlesPush(const real_t&);
template void ntt::PIC<ntt::Dim2>::ParticlesPush(const real_t&);
template void ntt::PIC<ntt::Dim3>::ParticlesPush(const real_t&);