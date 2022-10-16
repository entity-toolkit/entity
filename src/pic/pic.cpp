#include "wrapper.h"
#include "timer.h"
#include "pic.h"
#include "sim_params.h"

#include <plog/Log.h>

namespace ntt {

  template <Dimension D>
  void PIC<D>::Run() {
    Simulation<D, TypePIC>::Initialize();
    PLOGI << "... simulation initialized";
    Simulation<D, TypePIC>::InitializeSetup();
    PLOGI << "... simulation setup initialized";
    Simulation<D, TypePIC>::Verify();
    PLOGI << "... simulation verified";
    Simulation<D, TypePIC>::PrintDetails();
    PLOGI << "... simulation details printed";
    {
      auto  params = *(this->params());
      auto& mblock = this->meshblock;
      auto  timax  = (unsigned long)(params.totalRuntime() / mblock.timestep());
      PLOGI << "simulation mainloop started ...";

      FieldsExchange();
      FieldsBoundaryConditions();
      for (unsigned long ti {0}; ti < timax; ++ti) {
        PLOGD << "t = " << this->m_time;
        PLOGD << "ti = " << this->m_tstep;
        StepForward();
      }
      WaitAndSynchronize();
    }
    PLOGI << "... mainloop finished";
    Simulation<D, TypePIC>::Finalize();
    PLOGI << "... simulation finalized";
  }

  template <Dimension D>
  void PIC<D>::Benchmark() {
    Faraday();
    Ampere();
    Faraday();
  }

  template <Dimension D>
  void PIC<D>::StepForward() {
    timer::Timers timers({"Field_Solver",
                          "Field_BC",
                          "Curr_Deposit",
                          "Prtl_Pusher",
                          "Prtl_BC",
                          "User",
                          "Output"});
    auto          params = *(this->params());
    auto&         mblock = this->meshblock;
    auto&         wrtr   = this->writer;
    auto&         pgen   = this->problem_generator;

    if (params.fieldsolverEnabled()) {
      timers.start("Field_Solver");
      Faraday();
      timers.stop("Field_Solver");

      timers.start("Field_BC");
      FieldsExchange();
      FieldsBoundaryConditions();
      timers.stop("Field_BC");
    }

    {
      timers.start("Prtl_Pusher");
      ParticlesPush();
      timers.stop("Prtl_Pusher");

      timers.start("User");
      pgen.UserDriveParticles(this->m_time, params, mblock);
      timers.stop("User");

      timers.start("Prtl_BC");
      ParticlesBoundaryConditions();
      timers.stop("Prtl_BC");

      if (params.depositEnabled()) {
        timers.start("Curr_Deposit");
        ResetCurrents();
        CurrentsDeposit();

        timers.start("Field_BC");
        CurrentsSynchronize();
        CurrentsExchange();
        CurrentsBoundaryConditions();
        timers.stop("Field_BC");

        CurrentsFilter();
        timers.stop("Curr_Deposit");
      }

      timers.start("Prtl_BC");
      ParticlesExchange();
      timers.stop("Prtl_BC");
    }

    if (params.fieldsolverEnabled()) {
      timers.start("Field_Solver");
      Faraday();
      timers.stop("Field_Solver");

      timers.start("Field_BC");
      FieldsExchange();
      FieldsBoundaryConditions();
      timers.stop("Field_BC");

      timers.start("Field_Solver");
      Ampere();
      timers.stop("Field_Solver");

      if (params.depositEnabled()) {
        timers.start("Curr_Deposit");
        AmpereCurrents();
        timers.stop("Curr_Deposit");
      }

      timers.start("Field_BC");
      FieldsExchange();
      FieldsBoundaryConditions();
      timers.stop("Field_BC");
    }

    timers.start("Output");
    if (this->m_tstep % params.outputInterval() == 0) {
      if (params.outputFormat() != "disabled") {
        WaitAndSynchronize();
        ComputeDensity();
        this->SynchronizeHostDevice();
        ConvertFieldsToHat_h();
        wrtr.WriteFields(mblock, this->m_time, this->m_tstep);
      }
    }
    timers.stop("Output");

    timers.printAll("time = " + std::to_string(this->m_time)
                    + " : timestep = " + std::to_string(this->m_tstep));

    this->m_time += mblock.timestep();
    this->m_tstep++;
  }

  template <Dimension D>
  void PIC<D>::StepBackward() {}
} // namespace ntt

template class ntt::PIC<ntt::Dim1>;
template class ntt::PIC<ntt::Dim2>;
template class ntt::PIC<ntt::Dim3>;
