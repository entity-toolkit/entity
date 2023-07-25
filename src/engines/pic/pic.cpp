#include "pic.h"

#include "wrapper.h"

#include "sim_params.h"
#include "simulation.h"

#include "io/output.h"
#include "utils/timer.h"

#include <plog/Log.h>

namespace ntt {

  template <Dimension D>
  void PIC<D>::Run() {
    // register the content of em fields
    Simulation<D, PICEngine>::Verify();
    {
      auto  params = *(this->params());
      auto& mblock = this->meshblock;
      auto  timax  = (unsigned long)(params.totalRuntime() / mblock.timestep());

      ResetSimulation();
      Simulation<D, PICEngine>::PrintDetails();
      InitialStep();
      for (unsigned long ti { 0 }; ti < timax; ++ti) {
        PLOGV_(LogFile) << "step = " << this->m_tstep;
        PLOGV_(LogFile) << std::endl;
        StepForward();
      }
      WaitAndSynchronize();
    }
    Simulation<D, PICEngine>::Finalize();
  }

  template <Dimension D>
  void PIC<D>::InitializeSetup() {
    auto  params = *(this->params());
    auto& mblock = this->meshblock;
    problem_generator.UserInitFields(params, mblock);
    problem_generator.UserInitParticles(params, mblock);
  }

  template <Dimension D>
  void PIC<D>::InitialStep() {}

  template <Dimension D>
  void PIC<D>::Benchmark() {
    Faraday();
    Ampere();
    Faraday();
  }

  template <Dimension D>
  void PIC<D>::StepForward(const DiagFlags diag_flags) {
    NTTLog();
    const auto                      params     = *(this->params());
    const auto                      metadomain = *(this->metadomain());
    auto&                           mblock     = this->meshblock;
    auto&                           wrtr       = this->writer;
    auto&                           pgen       = this->problem_generator;

    timer::Timers                   timers({ "FieldSolver",
                                             "FieldBoundaries",
                                             "CurrentDeposit",
                                             "ParticlePusher",
                                             "ParticleBoundaries",
                                             "UserSpecific",
                                             "Output" },
                         params.blockingTimers());
    static std::vector<long double> tstep_durations = {};

    if (params.fieldsolverEnabled()) {
      timers.start("FieldSolver");
      Faraday();
      timers.stop("FieldSolver");

      timers.start("FieldBoundaries");
      this->Communicate(Comm_E);
      FieldsBoundaryConditions();
      timers.stop("FieldBoundaries");
    }

    {
      timers.start("ParticlePusher");
      ParticlesPush();
      timers.stop("ParticlePusher");

      timers.start("UserSpecific");
      pgen.UserDriveParticles(this->m_time, params, mblock);
      timers.stop("UserSpecific");

      timers.start("ParticleBoundaries");
      this->ParticlesBoundaryConditions();
      timers.stop("ParticleBoundaries");

      if (params.depositEnabled()) {
        timers.start("CurrentDeposit");
        CurrentsDeposit();

        timers.start("FieldBoundaries");
        this->CurrentsSynchronize();
        this->Communicate(Comm_J);
        CurrentsBoundaryConditions();
        timers.stop("FieldBoundaries");

        CurrentsFilter();
        timers.stop("CurrentDeposit");
      }

      timers.start("ParticleBoundaries");
      this->Communicate(Comm_Prtl);
      timers.stop("ParticleBoundaries");
    }

    if (params.fieldsolverEnabled()) {
      timers.start("FieldSolver");
      Faraday();
      timers.stop("FieldSolver");

      timers.start("FieldBoundaries");
      this->Communicate(Comm_B);
      FieldsBoundaryConditions();
      timers.stop("FieldBoundaries");

      timers.start("FieldSolver");
      Ampere();
      timers.stop("FieldSolver");

      if (params.depositEnabled()) {
        AmpereCurrents();
      }

      timers.start("FieldBoundaries");
      this->Communicate(Comm_E);
      FieldsBoundaryConditions();
      timers.stop("FieldBoundaries");
    }

    timers.start("Output");
    wrtr.WriteAll(params, metadomain, mblock, this->m_time, this->m_tstep);
    timers.stop("Output");

    this->PrintDiagnostics(this->m_tstep, this->m_time, timers, tstep_durations, diag_flags);

    this->m_time += mblock.timestep();
    pgen.setTime(this->m_time);
    this->m_tstep++;
  }

  template <Dimension D>
  void PIC<D>::StepBackward() {}
}    // namespace ntt

template class ntt::PIC<ntt::Dim1>;
template class ntt::PIC<ntt::Dim2>;
template class ntt::PIC<ntt::Dim3>;
