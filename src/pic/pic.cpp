#include "global.h"
#include "timer.h"
#include "pic.h"
#include "sim_params.h"

namespace ntt {

  template <Dimension D>
  void PIC<D>::mainloop() {
    unsigned long timax {static_cast<unsigned long>(this->m_sim_params.total_runtime()
                                                    / this->m_mblock.timestep())};
    real_t        time {0.0};
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
    this->mainloop();
    PLOGD << "<<< simulation mainloop finished.";
    this->finalize();
    PLOGD << "Simulation finalized.";
  }

  template <Dimension D>
  void PIC<D>::step_forward(const real_t& time) {
    TimerCollection timers(
      {"Field_Solver", "Field_BC", "Curr_Deposit", "Prtl_Pusher", "Prtl_BC"});
    if (this->sim_params().enable_fieldsolver()) {
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

      if (this->sim_params().enable_deposit()) {
        timers.start(3);
        resetCurrents(time);
        depositCurrentsSubstep(time);
        filterCurrentsSubstep(time);
        transformCurrentsSubstep(time);
        timers.stop(3);
      }

      timers.start(5);
      particleBoundaryConditions(time);
      timers.stop(5);
    }

    if (this->sim_params().enable_fieldsolver()) {
      timers.start(1);
      faradaySubstep(time, HALF);
      timers.stop(1);

      timers.start(2);
      fieldBoundaryConditions(time);
      timers.stop(2);

      timers.start(1);
      ampereSubstep(time, ONE);
      timers.stop(1);

      if (this->sim_params().enable_deposit()) {
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
