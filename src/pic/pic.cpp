#if SIMTYPE == PIC_SIMTYPE

#  include "global.h"
#  include "timer.h"
#  include "pic.h"

namespace ntt {

  template <Dimension D>
  void PIC<D>::mainloop() {
    unsigned long timax {static_cast<unsigned long>(this->m_sim_params.total_runtime() / this->m_mblock.timestep())};
    real_t time {0.0};
    fieldBoundaryConditions(ZERO);
    for (unsigned long ti {0}; ti < timax; ++ti) {
      PLOGD << "t = " << time;
      step_forward(time);
      time += this->m_mblock.timestep();
    }
  }

  template <Dimension D>
  void PIC<D>::process() {
    this->initialize();
    PLOGD << "Simulation initialized.";
    this->initializeSetup();
    PLOGD << "Setup initialized.";
    this->verify();
    PLOGD << "Prerun check passed.";
    this->printDetails();
    PLOGD << "Simulation details printed.";

    PLOGD << "Simulation mainloop started >>>";
    mainloop();
    PLOGD << "<<< simulation mainloop finished.";
    this->finalize();
    PLOGD << "Simulation finalized.";
  }

  template <Dimension D>
  void PIC<D>::step_forward(const real_t& time) {
    TimerCollection timers({"Field_Solver", "Field_BC", "Curr_Deposit", "Prtl_Pusher"});
    {
      timers.start(1);
      faradaySubstep(time, 1.0);
      timers.stop(1);
    }

    {
      timers.start(2);
      fieldBoundaryConditions(time);
      timers.stop(2);
    }

    {
      timers.start(1);
      ampereSubstep(time, 1.0);
      timers.stop(1);
    }

    {
      timers.start(2);
      fieldBoundaryConditions(time);
      timers.stop(2);
    }
    timers.printAll(millisecond);
  }

  template <Dimension D>
  void PIC<D>::step_backward(const real_t& time) {
    TimerCollection timers({"Field_Solver", "Field_BC", "Curr_Deposit", "Prtl_Pusher"});
    {
      timers.start(1);
      ampereSubstep(time, -1.0);
      timers.stop(1);
    }

    {
      timers.start(2);
      fieldBoundaryConditions(time);
      timers.stop(2);
    }

    {
      timers.start(1);
      faradaySubstep(time, -1.0);
      timers.stop(1);
    }

    {
      timers.start(2);
      fieldBoundaryConditions(time);
      timers.stop(2);
    }
    timers.printAll(millisecond);
  }
} // namespace ntt

template class ntt::PIC<ntt::Dimension::ONE_D>;
template class ntt::PIC<ntt::Dimension::TWO_D>;
template class ntt::PIC<ntt::Dimension::THREE_D>;

// {
//   timers.start(1);
//   faradaySubstep(time, 0.5);
//   timers.stop(1);
// }

// {
//   timers.start(2);
//   fieldBoundaryConditions(time);
//   timers.stop(2);
// }

// // {
// //   timers.start(4);
// //   pushParticlesSubstep(time);
// //   timers.stop(4);
// // }

// // depositSubstep(time);

// // {
// //   timers.start(2);
// //   particleBoundaryConditions(time);
// //   timers.stop(2);
// // }
// // BC currents

// {
//   timers.start(1);
//   faradaySubstep(time, 0.5);
//   timers.stop(1);
// }

// {
//   timers.start(2);
//   fieldBoundaryConditions(time);
//   timers.stop(2);
// }

// {
//   timers.start(1);
//   ampereSubstep(time, 1.0);
//   addCurrentsSubstep(time);
//   resetCurrentsSubstep(time);
//   timers.stop(1);
// }

// {
//   timers.start(2);
//   fieldBoundaryConditions(time);
//   timers.stop(2);
// }

#endif