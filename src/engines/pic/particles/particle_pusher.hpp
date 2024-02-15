#ifndef PIC_PARTICLE_PUSHER_HPP
#define PIC_PARTICLE_PUSHER_HPP

#include "meshblock/meshblock.h"
#include "meshblock/particles.h"

#include "kernels/particle_pusher_sr.hpp"

namespace ntt {

  template <Dimension D, class M, class PG, typename P, bool ExtForce>
  void PushLoopWith(const SimulationParams&  params,
                    Meshblock<D, PICEngine>& mblock,
                    Particles<D, PICEngine>& species,
                    PG&                      pgen,
                    real_t                   time,
                    real_t                   factor) {
    const auto dt              = factor * mblock.timestep();
    const auto charge_ovr_mass = species.mass() > ZERO
                                   ? species.charge() / species.mass()
                                   : ZERO;
    const auto coeff           = charge_ovr_mass * HALF * dt * params.omegaB0();
    if (species.cooling() == Cooling::NONE) {
      Kokkos::parallel_for(
        "ParticlesPush",
        Kokkos::RangePolicy<AccelExeSpace, P>(0, species.npart()),
        Pusher_kernel<D, M, PG, P, ExtForce, NoCooling_t>(mblock.em,
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
                                                          pgen,
                                                          time,
                                                          coeff,
                                                          dt,
                                                          mblock.Ni1(),
                                                          mblock.Ni2(),
                                                          mblock.Ni3(),
                                                          mblock.boundaries,
                                                          params.GCALarmorMax(),
                                                          params.GCAEovrBMax(),
                                                          ZERO));
    } else if (species.cooling() == Cooling::SYNCHROTRON) {
      const auto coeff_sync = (real_t)(0.1) * dt * params.omegaB0() /
                              (SQR(params.SynchrotronGammarad()) * species.mass());
      Kokkos::parallel_for(
        "ParticlesPush",
        Kokkos::RangePolicy<AccelExeSpace, P>(0, species.npart()),
        Pusher_kernel<D, M, PG, P, ExtForce, Synchrotron_t>(mblock.em,
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
                                                            pgen,
                                                            time,
                                                            coeff,
                                                            dt,
                                                            mblock.Ni1(),
                                                            mblock.Ni2(),
                                                            mblock.Ni3(),
                                                            mblock.boundaries,
                                                            params.GCALarmorMax(),
                                                            params.GCAEovrBMax(),
                                                            coeff_sync));
    }
  }

  template <Dimension D, class M, class PG, bool ExtForce>
  void PushLoop(const SimulationParams&  params,
                Meshblock<D, PICEngine>& mblock,
                Particles<D, PICEngine>& species,
                PG&                      pgen,
                real_t                   time,
                real_t                   factor) {
    const auto pusher = species.pusher();
    if (pusher == ParticlePusher::PHOTON) {
      PushLoopWith<D, M, PG, Photon_t, ExtForce>(params,
                                                 mblock,
                                                 species,
                                                 pgen,
                                                 time,
                                                 factor);
    } else if (pusher == ParticlePusher::BORIS) {
      PushLoopWith<D, M, PG, Boris_t, ExtForce>(params, mblock, species, pgen, time, factor);
    } else if (pusher == ParticlePusher::VAY) {
      PushLoopWith<D, M, PG, Vay_t, ExtForce>(params, mblock, species, pgen, time, factor);
    } else if (pusher == ParticlePusher::BORIS_GCA) {
      PushLoopWith<D, M, PG, Boris_GCA_t, ExtForce>(params,
                                                    mblock,
                                                    species,
                                                    pgen,
                                                    time,
                                                    factor);
    } else if (pusher == ParticlePusher::VAY_GCA) {
      PushLoopWith<D, M, PG, Vay_GCA_t, ExtForce>(params,
                                                  mblock,
                                                  species,
                                                  pgen,
                                                  time,
                                                  factor);
    } else {
      NTTHostError("not implemented");
    }
  }
} // namespace ntt

#endif // PIC_PARTICLE_PUSHER_HPP