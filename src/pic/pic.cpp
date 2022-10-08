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
        Simulation<D, TypePIC>::WriteOutput(ti);
      }
      WaitAndSynchronize();
    }
    PLOGI << "... mainloop finished";
    // this->benchmark();
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
    TimerCollection timers(
      {"Field_Solver", "Field_BC", "Curr_Deposit", "Prtl_Pusher", "Prtl_BC", "User"});
    auto  params = *(this->params());
    auto& mblock = this->meshblock;
    auto& pgen   = this->problem_generator;

    if (params.fieldsolverEnabled()) {
      timers.start(1);
      Faraday();
      timers.stop(1);

      timers.start(2);
      FieldsExchange();
      FieldsBoundaryConditions();
      timers.stop(2);
    }

    {
      timers.start(4);
      ParticlesPush();
      timers.stop(4);

      timers.start(6);
      pgen.UserDriveParticles(this->m_time, params, mblock);
      timers.stop(6);

      timers.start(5);
      ParticlesBoundaryConditions();
      timers.stop(5);

      if (params.depositEnabled()) {
        timers.start(3);
        ResetCurrents();
        CurrentsDeposit();

        timers.start(2);
        CurrentsSynchronize();
        CurrentsExchange();
        CurrentsBoundaryConditions();
        timers.stop(2);

        CurrentsFilter();
        timers.stop(3);
      }

      timers.start(5);
      ParticlesExchange();
      timers.stop(5);
    }

    if (params.fieldsolverEnabled()) {
      timers.start(1);
      Faraday();
      timers.stop(1);

      timers.start(2);
      FieldsExchange();
      FieldsBoundaryConditions();
      timers.stop(2);

      timers.start(1);
      Ampere();
      timers.stop(1);

      if (params.depositEnabled()) {
        timers.start(3);
        AmpereCurrents();
        timers.stop(3);
      }

      timers.start(2);
      FieldsExchange();
      FieldsBoundaryConditions();
      timers.stop(2);
    }
    timers.printAll(millisecond);

    this->m_time += mblock.timestep();
    this->m_tstep++;
  }

  template <Dimension D>
  void PIC<D>::StepBackward() {}
} // namespace ntt

template struct ntt::PIC<ntt::Dim1>;
template struct ntt::PIC<ntt::Dim2>;
template struct ntt::PIC<ntt::Dim3>;
