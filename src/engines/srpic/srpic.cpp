#include "engines/srpic/srpic.h"

#include "enums.h"

#include "arch/traits.h"
#include "utils/log.h"
#include "utils/timer.h"

#include "metrics/minkowski.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "framework/domain/domain.h"

#include <string>

namespace ntt {

  template <class M>
  void SRPICEngine<M>::step_forward(timer::Timers& timers, domain_t& dom) {
    const auto fieldsolver_enabled = m_params.template get<bool>(
      "algorithms.toggles.fieldsolver");
    const auto deposit_enabled = m_params.template get<bool>(
      "algorithms.toggles.deposit");

    if (fieldsolver_enabled) {
      timers.start("FieldSolver");
      Faraday(dom, HALF);
      timers.stop("FieldSolver");

      timers.start("Communications");
      // !TODO communicate B-field
      timers.stop("Communications");

      timers.start("FieldBoundaries");
      // !TODO field boundaries
      timers.stop("FieldBoundaries");
    }

    {
      timers.start("ParticlePusher");
      ParticlePush(dom);
      timers.stop("ParticlePusher");

      if (deposit_enabled) {
        timers.start("CurrentDeposit");
        CurrentsDeposit(dom);
        timers.stop("CurrentDeposit");

        timers.start("Communications");
        // !TODO communicate currents
        timers.stop("Communications");

        timers.start("CurrentFiltering");
        // !TODO filtering
        timers.stop("CurrentFiltering");
      }

      timers.start("Communications");
      // !TODO communicate particles
      timers.stop("Communications");
    }

    if (fieldsolver_enabled) {
      timers.start("FieldSolver");
      Faraday(dom, HALF);
      timers.stop("FieldSolver");

      timers.start("Communications");
      // !TODO communicate B-field
      timers.stop("Communications");

      timers.start("FieldBoundaries");
      // !TODO field boundaries
      timers.stop("FieldBoundaries");

      timers.start("FieldSolver");
      Ampere(dom, ONE);
      timers.stop("FieldSolver");

      if (deposit_enabled) {
        timers.start("FieldSolver");
        CurrentsAmpere(dom);
        timers.stop("FieldSolver");
      }

      timers.start("Communications");
      // !TODO communicate E-field
      timers.stop("Communications");

      timers.start("FieldBoundaries");
      // !TODO field boundaries
      timers.stop("FieldBoundaries");
    }
  }

  template class SRPICEngine<metric::Minkowski<Dim::_1D>>;
  template class SRPICEngine<metric::Minkowski<Dim::_2D>>;
  template class SRPICEngine<metric::Minkowski<Dim::_3D>>;
  template class SRPICEngine<metric::Spherical<Dim::_2D>>;
  template class SRPICEngine<metric::QSpherical<Dim::_2D>>;

  // if (params.fieldsolverEnabled()) {
  //   timers.start("FieldSolver");
  //   Faraday();
  //   timers.stop("FieldSolver");

  //   { mblock.CheckNaNs("After 1st Faraday", CheckNaN_Fields); }

  //   timers.start("Communications");
  //   this->Communicate(Comm_E);
  //   timers.stop("Communications");

  //   timers.start("FieldBoundaries");
  //   FieldsBoundaryConditions();
  //   timers.stop("FieldBoundaries");

  //   { mblock.CheckNaNs("After 1st Fields BC", CheckNaN_Fields); }
  // }

  // {
  //   timers.start("ParticlePusher");
  //   ParticlesPush();
  //   timers.stop("ParticlePusher");

  //   { mblock.CheckNaNs("After Push", CheckNaN_Particles); }

  //   if (params.depositEnabled()) {
  //     timers.start("CurrentDeposit");
  //     CurrentsDeposit();
  //     timers.stop("CurrentDeposit");

  //     { mblock.CheckNaNs("After Deposit", CheckNaN_Currents); }

  //     timers.start("Communications");
  //     this->CurrentsSynchronize();
  //     timers.stop("Communications");

  //     { mblock.CheckNaNs("After Currents BC", CheckNaN_Currents); }

  //     timers.start("CurrentFiltering");
  //     CurrentsFilter();
  //     timers.stop("CurrentFiltering");

  //     { mblock.CheckNaNs("After Currents Filter", CheckNaN_Currents); }
  //   }

  //   timers.start("Communications");
  //   this->Communicate(Comm_Prtl);
  //   timers.stop("Communications");

  //   {
  //     mblock.CheckNaNs("After Prtls Comm", CheckNaN_Particles);
  //     mblock.CheckOutOfBounds("After Prtls Comm");
  //   }
  // }

  // if (params.fieldsolverEnabled()) {
  //   timers.start("FieldSolver");
  //   Faraday();
  //   timers.stop("FieldSolver");

  //   timers.start("Communications");
  //   this->Communicate(Comm_B);
  //   timers.stop("Communications");

  //   timers.start("FieldBoundaries");
  //   FieldsBoundaryConditions();
  //   timers.stop("FieldBoundaries");

  //   { mblock.CheckNaNs("After 2nd Fields BC", CheckNaN_Fields); }

  //   timers.start("FieldSolver");
  //   Ampere();
  //   timers.stop("FieldSolver");

  //   { mblock.CheckNaNs("After Ampere", CheckNaN_Fields); }

  //   if (params.depositEnabled()) {
  //     timers.start("FieldSolver");
  //     AmpereCurrents();
  //     timers.stop("FieldSolver");

  //     { mblock.CheckNaNs("After Ampere Currents", CheckNaN_Fields); }
  //   }

  //   timers.start("Communications");
  //   this->Communicate(Comm_E);
  //   timers.stop("Communications");

  //   timers.start("FieldBoundaries");
  //   FieldsBoundaryConditions();
  //   timers.stop("FieldBoundaries");

  //   { mblock.CheckNaNs("After 3rd Fields BC", CheckNaN_Fields); }
  // }

  // timers.start("UserSpecific");
  // pgen.UserDriveParticles(this->m_time, params, mblock);
  // timers.stop("UserSpecific");

  // { mblock.CheckNaNs("After Drive", CheckNaN_Particles); }

  // timers.start("Output");
  // wrtr.WriteAll(params, metadomain, mblock, this->m_time, this->m_tstep);
  // timers.stop("Output");

  // this->PrintDiagnostics(this->m_tstep,
  //                        this->m_time,
  //                        timers,
  //                        tstep_durations,
  //                        diag_flags);

  // this->m_time += mblock.timestep();
  // pgen.setTime(this->m_time);
  // this->m_tstep++;

} // namespace ntt