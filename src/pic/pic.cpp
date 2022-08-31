#include "global.h"
#include "timer.h"
#include "pic.h"
#include "sim_params.h"

namespace ntt {

  template <Dimension D>
  void PIC<D>::process() {
    this->initialize();
    this->initializeSetup();
    this->verify();
    this->printDetails();
    this->mainloop();
    // this->benchmark();
    this->finalize();
  }

  template <Dimension D>
  void PIC<D>::benchmark() {
    faradaySubstep(0.0, HALF);
    ampereSubstep(0.0, ONE);
    faradaySubstep(0.0, HALF);
  }

  template <Dimension D>
  void PIC<D>::mainloop() {
    unsigned long timax {static_cast<unsigned long>(this->m_sim_params.total_runtime()
                                                    / this->m_mblock.timestep())};
    real_t        time {0.0};
    PLOGD << "Simulation mainloop started >>>";

    fieldBoundaryConditions(ZERO);
    for (unsigned long ti {0}; ti < timax; ++ti) {
      PLOGD << "t = " << time;
      step_forward(time);
      this->writeOutput(ti);
      time += this->m_mblock.timestep();
    }
    WaitAndSynchronize();
    PLOGD << "<<< simulation mainloop finished.";
  }

  template <Dimension D>
  void PIC<D>::step_forward(const real_t& time) {
    TimerCollection timers(
      {"Field_Solver", "Field_BC", "Curr_Deposit", "Prtl_Pusher", "Prtl_BC"});
    if (this->sim_params()->enable_fieldsolver()) {
      timers.start(1);
      faradaySubstep(time, HALF);
      timers.stop(1);

      timers.start(2);
      fieldBoundaryConditions(time);
      timers.stop(2);
    }

    {
      timers.start(4);
      pushParticlesSubstep(time, ONE);
      timers.stop(4);

      if (this->sim_params()->enable_deposit()) {
        timers.start(3);
        resetCurrents(time);
        depositCurrentsSubstep(time);

        timers.start(2);
        currentBoundaryConditions(time);
        timers.stop(2);

        filterCurrentsSubstep(time);
        timers.stop(3);
      }

      timers.start(5);
      particleBoundaryConditions(time);
      timers.stop(5);
    }

    if (this->sim_params()->enable_fieldsolver()) {
      timers.start(1);
      faradaySubstep(time, HALF);
      timers.stop(1);

      timers.start(2);
      fieldBoundaryConditions(time);
      timers.stop(2);

      timers.start(1);
      ampereSubstep(time, ONE);
      timers.stop(1);

      if (this->sim_params()->enable_deposit()) {
        timers.start(3);
        addCurrentsSubstep(time);
        timers.stop(3);
      }

      timers.start(2);
      fieldBoundaryConditions(time);
      timers.stop(2);
    }
    timers.printAll(millisecond);
  }

  template <Dimension D>
  void PIC<D>::step_backward(const real_t&) {}
} // namespace ntt

template class ntt::PIC<ntt::Dimension::ONE_D>;
template class ntt::PIC<ntt::Dimension::TWO_D>;
template class ntt::PIC<ntt::Dimension::THREE_D>;
