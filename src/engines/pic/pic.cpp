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
    auto  params = *(this->params());
    auto& mblock = this->meshblock;
    auto  timax  = (unsigned long)(params.totalRuntime() / mblock.timestep());

    ResetSimulation();
    this->PrintDetails();
    this->Verify();
    InitialStep();
    for (unsigned long ti { 0 }; ti < timax; ++ti) {
      NTTLogPrint(fmt::format("step = %d", this->m_tstep));
      StepForward();
    }
    WaitAndSynchronize();
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
    const auto params     = *(this->params());
    const auto metadomain = *(this->metadomain());
    auto&      mblock     = this->meshblock;
    auto&      wrtr       = this->writer;
    auto&      pgen       = this->problem_generator;

    timer::Timers                   timers({ "FieldSolver",
                                             "CurrentFiltering",
                                             "CurrentDeposit",
                                             "ParticlePusher",
                                             "FieldBoundaries",
                                             "ParticleBoundaries",
                                             "Communications",
                                             "UserSpecific",
                                             "Output" },
                         params.blockingTimers());
    static std::vector<long double> tstep_durations = {};

    if (params.fieldsolverEnabled()) {
      timers.start("FieldSolver");
      Faraday();
      timers.stop("FieldSolver");

      { mblock.CheckNaNs("After 1st Faraday", CheckNaN_Fields); }

      timers.start("Communications");
      this->Communicate(Comm_E);
      timers.stop("Communications");

      timers.start("FieldBoundaries");
      FieldsBoundaryConditions();
      timers.stop("FieldBoundaries");

      { mblock.CheckNaNs("After 1st Fields BC", CheckNaN_Fields); }
    }

    {
      timers.start("ParticlePusher");
      ParticlesPush();
      timers.stop("ParticlePusher");

      {
        mblock.CheckNaNs("After Push", CheckNaN_Particles);
      }

      if (params.depositEnabled()) {
        timers.start("CurrentDeposit");
        CurrentsDeposit();
        timers.stop("CurrentDeposit");

        { mblock.CheckNaNs("After Deposit", CheckNaN_Currents); }

        timers.start("Communications");
        this->CurrentsSynchronize();
        timers.stop("Communications");

        { mblock.CheckNaNs("After Currents BC", CheckNaN_Currents); }

        timers.start("CurrentFiltering");
        CurrentsFilter();
        timers.stop("CurrentFiltering");

        { mblock.CheckNaNs("After Currents Filter", CheckNaN_Currents); }
      }

      timers.start("Communications");
      this->Communicate(Comm_Prtl);
      timers.stop("Communications");

      {
        mblock.CheckNaNs("After Prtls Comm", CheckNaN_Particles);
        mblock.CheckOutOfBounds("After Prtls Comm");
      }
    }

    if (params.fieldsolverEnabled()) {
      timers.start("FieldSolver");
      Faraday();
      timers.stop("FieldSolver");

      timers.start("Communications");
      this->Communicate(Comm_B);
      timers.stop("Communications");

      timers.start("FieldBoundaries");
      FieldsBoundaryConditions();
      timers.stop("FieldBoundaries");

      { mblock.CheckNaNs("After 2nd Fields BC", CheckNaN_Fields); }

      timers.start("FieldSolver");
      Ampere();
      timers.stop("FieldSolver");

      { mblock.CheckNaNs("After Ampere", CheckNaN_Fields); }

      if (params.depositEnabled()) {
        timers.start("FieldSolver");
        AmpereCurrents();
        timers.stop("FieldSolver");

        { mblock.CheckNaNs("After Ampere Currents", CheckNaN_Fields); }
      }

      timers.start("Communications");
      this->Communicate(Comm_E);
      timers.stop("Communications");

      timers.start("FieldBoundaries");
      FieldsBoundaryConditions();
      timers.stop("FieldBoundaries");

      { mblock.CheckNaNs("After 3rd Fields BC", CheckNaN_Fields); }
    }

    timers.start("UserSpecific");
    pgen.UserDriveParticles(this->m_time, params, mblock);
    timers.stop("UserSpecific");

    { mblock.CheckNaNs("After Drive", CheckNaN_Particles); }

    timers.start("Output");
    wrtr.WriteAll(params, metadomain, mblock, this->m_time, this->m_tstep);
    timers.stop("Output");

    this->PrintDiagnostics(this->m_tstep,
                           this->m_time,
                           timers,
                           tstep_durations,
                           diag_flags);

    this->m_time += mblock.timestep();
    pgen.setTime(this->m_time);
    this->m_tstep++;
  }

  template <Dimension D>
  void PIC<D>::StepBackward() {}
} // namespace ntt

template class ntt::PIC<ntt::Dim1>;
template class ntt::PIC<ntt::Dim2>;
template class ntt::PIC<ntt::Dim3>;
