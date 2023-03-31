#include "pic.h"

#include "wrapper.h"

#include "fields.h"
#include "progressbar.h"
#include "sim_params.h"
#include "timer.h"

#include <plog/Log.h>

namespace ntt {

  template <Dimension D>
  void PIC<D>::Run() {
    // register the content of em fields
    Simulation<D, PICEngine>::Initialize();
    Simulation<D, PICEngine>::Verify();
    {
      auto  params = *(this->params());
      auto& mblock = this->meshblock;
      auto  timax  = (unsigned long)(params.totalRuntime() / mblock.timestep());

      ResetSimulation();
      Simulation<D, PICEngine>::PrintDetails();
      InitialStep();
      for (unsigned long ti { 0 }; ti < timax; ++ti) {
        PLOGI_(LogFile) << "ti " << this->m_tstep << "...";
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
  void PIC<D>::InitialStep() {
    auto& mblock = this->meshblock;
    ImposeContent(mblock.em_content,
                  { Content::ex1_cntrv,
                    Content::ex2_cntrv,
                    Content::ex3_cntrv,
                    Content::bx1_cntrv,
                    Content::bx2_cntrv,
                    Content::bx3_cntrv });
  }

  template <Dimension D>
  void PIC<D>::Benchmark() {
    Faraday();
    Ampere();
    Faraday();
  }

  template <Dimension D>
  void PIC<D>::StepForward() {
    NTTLog();
    auto                            params = *(this->params());
    auto&                           mblock = this->meshblock;
    auto&                           wrtr   = this->writer;
    auto&                           pgen   = this->problem_generator;

    timer::Timers                   timers({ "FieldSolver",
                                             "FieldBoundaries",
                                             "CurrentDeposit",
                                             "ParticlePusher",
                                             "ParticleBoundaries",
                                             "UserSpecific",
                                             "Output" });
    static std::vector<double>      dead_fractions  = {};
    static std::vector<long double> tstep_durations = {};

    if (params.fieldsolverEnabled()) {
      timers.start("FieldSolver");
      Faraday();
      timers.stop("FieldSolver");

      timers.start("FieldBoundaries");
      FieldsExchange();
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
      ParticlesBoundaryConditions();
      timers.stop("ParticleBoundaries");

      if (params.depositEnabled()) {
        timers.start("CurrentDeposit");
        ResetCurrents();
        CurrentsDeposit();

        timers.start("FieldBoundaries");
        CurrentsSynchronize();
        CurrentsExchange();
        CurrentsBoundaryConditions();
        timers.stop("FieldBoundaries");

        CurrentsFilter();
        timers.stop("CurrentDeposit");
      }

      timers.start("ParticleBoundaries");
      ParticlesExchange();
      if ((params.shuffleInterval() > 0) && (this->m_tstep % params.shuffleInterval() == 0)) {
        dead_fractions = mblock.RemoveDeadParticles(params.maxDeadFraction());
      }
      timers.stop("ParticleBoundaries");
    }

    if (params.fieldsolverEnabled()) {
      timers.start("FieldSolver");
      Faraday();
      timers.stop("FieldSolver");

      timers.start("FieldBoundaries");
      FieldsExchange();
      FieldsBoundaryConditions();
      timers.stop("FieldBoundaries");

      timers.start("FieldSolver");
      Ampere();
      timers.stop("FieldSolver");

      if (params.depositEnabled()) {
        timers.start("CurrentDeposit");
        AmpereCurrents();
        timers.stop("CurrentDeposit");
      }

      timers.start("FieldBoundaries");
      FieldsExchange();
      FieldsBoundaryConditions();
      timers.stop("FieldBoundaries");
    }

    timers.start("Output");
    if ((params.outputFormat() != "disabled")
        && (this->m_tstep % params.outputInterval() == 0)) {
      WaitAndSynchronize();
      wrtr.WriteFields(params, mblock, this->m_time, this->m_tstep);
    }
    timers.stop("Output");

    timers.printAll("time = " + std::to_string(this->m_time)
                    + " : timestep = " + std::to_string(this->m_tstep));
    this->PrintDiagnostics(std::cout, dead_fractions);
    tstep_durations.push_back(timers.get("Total"));
    std::cout << std::setw(46) << std::setfill('-') << "" << std::endl;
    ProgressBar(tstep_durations, this->m_time, params.totalRuntime());
    std::cout << std::setw(46) << std::setfill('=') << "" << std::endl;

    ImposeEmptyContent(mblock.buff_content);
    ImposeEmptyContent(mblock.cur_content);
    ImposeEmptyContent(mblock.bckp_content);

    this->m_time += mblock.timestep();
    this->m_tstep++;
  }

  template <Dimension D>
  void PIC<D>::StepBackward() {}
}    // namespace ntt

template class ntt::PIC<ntt::Dim1>;
template class ntt::PIC<ntt::Dim2>;
template class ntt::PIC<ntt::Dim3>;
