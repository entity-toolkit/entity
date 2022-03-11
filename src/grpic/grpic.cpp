#include "global.h"
#include "timer.h"
#include "grpic.h"
#include "sim_params.h"

namespace ntt {

  template <Dimension D>
  void GRPIC<D>::mainloop() {
    unsigned long timax {static_cast<unsigned long>(this->m_sim_params.total_runtime() / this->m_mblock.timestep())};
    real_t time {0.0};
    fieldBoundaryConditions(ZERO, 0);
    initial_step(ZERO);
    for (unsigned long ti {0}; ti < timax; ++ti) {
      PLOGD << "t = " << time;
      step_forward(time);
      time += this->m_mblock.timestep();
    }
  }

  template <Dimension D>
  void GRPIC<D>::process() {
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
  void GRPIC<D>::initial_step(const real_t& time) {
    
    // Initially: B0, D0, B1, D1 at t=0
  
      // E, H at t=0
      Compute_E_Substep(time, 0);
      AuxiliaryBoundaryConditions(time, 0);
      Compute_H_Substep(time, 0);
      AuxiliaryBoundaryConditions(time, 1);

      // B0 at t=1/2
      faradaySubstep(time, 0.5, 0);
      fieldBoundaryConditions(time, 1);

      // D1 at t=1/2
      ampereSubstep(time, 0.5, -1);
      fieldBoundaryConditions(time, 0);

      // E, H at t=1/2
      Compute_E_Substep(time, 1);
      AuxiliaryBoundaryConditions(time, 0);
      Compute_H_Substep(time, 0);
      AuxiliaryBoundaryConditions(time, 1);

      // B0 at t=1
      faradaySubstep(time, 1.0, 1);
      fieldBoundaryConditions(time, 1);

      // D0 at t=1
      ampereSubstep(time, 1.0, 0);
      fieldBoundaryConditions(time, 0);

      // H at t=1
      Compute_H_Substep(time, 1);
      AuxiliaryBoundaryConditions(time, 1);

      // D0 at t=3/2
      ampereSubstep(time, 1.0, 1);
      fieldBoundaryConditions(time, 0);

      // Final: B0 at 0, D0 at 1/2, B1 at 1, D1 at 3/2  
      swap_em_cur(this->m_mblock);
  }

  template <Dimension D>
  void GRPIC<D>::step_forward(const real_t& time) {
    TimerCollection timers({"Field_solver", "Field_BC", "Curr_Deposit", "Prtl_Pusher"});
    
    // Initially: B0 at n-3/2, B at n-1/2, D0 at n-1, D at n, x at n, u at n-1/2, J0 at n-1, J at n-1/2 
    
      timers.start(1);
      // B0 at n-1, B at n-1/2, D0 at n-1/2, D at n
      Average_EM_Substep(time);
      // E at n-1/2 with B and D0
      Compute_E_Substep(time, 0);
      AuxiliaryBoundaryConditions(time, 0);
      // B0 at n, B at n-1/2 
      faradaySubstep(time, 1.0, 0);
      timers.stop(1);

      timers.start(2);
      fieldBoundaryConditions(time, 1);
      timers.stop(2);

      timers.start(1);
      // H at n with B0 and D
      Compute_H_Substep(time, 0);
      AuxiliaryBoundaryConditions(time, 1);
      timers.stop(1);

    // Push particles
    // x at n+1, u at n+1/2
      timers.start(4);
      timers.stop(4);

    // Current deposition
    // J0 at n+1/2, J at n-1/2
      timers.start(3);
      timers.stop(3);
    
      timers.start(1);
      // J0 at n+1/2, J at n
      Average_J_Substep(time);
      // E at n with B0 and D
      Compute_E_Substep(time, 1);
      AuxiliaryBoundaryConditions(time, 0);
      // B0 at n+1/2, B at n-1/2
      faradaySubstep(time, 1.0, 1);
      timers.stop(1);

      timers.start(2);
      fieldBoundaryConditions(time, 1);
      timers.stop(2);

      timers.start(1);
      // D0 at n+1/2, D at n
      ampereSubstep(time, 1.0, 0);
      timers.stop(1);

      timers.start(2);
      fieldBoundaryConditions(time, 0);
      timers.stop(2);

      timers.start(1);
      // H at n+1/2 with B0 and D0
      Compute_H_Substep(time, 1);
      AuxiliaryBoundaryConditions(time, 1);
      // D0 at n+1, D at n
      ampereSubstep(time, 1.0, 1);
      
      // Final: B0 at n-1/2, B at n+1/2, D0 at n, D at n+1, x at n+1, u at n+1/2, J0 at n, J at n+1/2
      swap_em_cur(this->m_mblock);
      timers.stop(1);
    
      timers.start(2);
      fieldBoundaryConditions(time, 0);
      timers.stop(2);
      
      timers.printAll(millisecond);
    
  }

  template <Dimension D>
  void GRPIC<D>::step_backward(const real_t& time) {
  TimerCollection timers({"Field_solver", "Field_BC", "Curr_Deposit", "Prtl_Pusher"});

   // Initially: B0 at n-3/2, B at n-1/2, D0 at n-1, D at n, x at n, u at n-1/2, J0 at n-1, J at n-1/2 
    
      timers.start(1);
      // B0 at n-1, B at n-1/2, D0 at n-1/2, D at n
      Average_EM_Substep(time);
      // E at n-1/2 with B and D0
      Compute_E_Substep(time, 0);
      AuxiliaryBoundaryConditions(time, 0);
      // B0 at n, B at n-1/2 
      faradaySubstep(time, - 1.0, 0);
      timers.stop(1);

      timers.start(2);
      fieldBoundaryConditions(time, 1);
      timers.stop(2);

      timers.start(1);
      // H at n with B0 and D
      Compute_H_Substep(time, 0);
      AuxiliaryBoundaryConditions(time, 1);
      timers.stop(1);

    // Push particles
    // x at n+1, u at n+1/2
      timers.start(4);
      timers.stop(4);

    // Current deposition
    // J0 at n+1/2, J at n-1/2
      timers.start(3);
      timers.stop(3);
    
      timers.start(1);
      // J0 at n+1/2, J at n
      Average_J_Substep(time);
      // E at n with B0 and D
      Compute_E_Substep(time, 1);
      AuxiliaryBoundaryConditions(time, 0);
      // B0 at n+1/2, B at n-1/2
      faradaySubstep(time, - 1.0, 1);
      timers.stop(1);

      timers.start(2);
      fieldBoundaryConditions(time, 1);
      timers.stop(2);

      timers.start(1);
      // D0 at n+1/2, D at n
      ampereSubstep(time, - 1.0, 0);
      timers.stop(1);

      timers.start(2);
      fieldBoundaryConditions(time, 0);
      timers.stop(2);

      timers.start(1);
      // H at n+1/2 with B0 and D0
      Compute_H_Substep(time, 1);
      AuxiliaryBoundaryConditions(time, 1);
      // D0 at n+1, D at n
      ampereSubstep(time, - 1.0, 1);
      
      // Final: B0 at n-1/2, B at n+1/2, D0 at n, D at n+1, x at n+1, u at n+1/2, J0 at n, J at n+1/2
      swap_em_cur(this->m_mblock);
      timers.stop(1);
    
      timers.start(2);
      fieldBoundaryConditions(time, 0);
      timers.stop(2);
      
      timers.printAll(millisecond);
      
   }
} // namespace ntt

template class ntt::GRPIC<ntt::Dimension::TWO_D>;
template class ntt::GRPIC<ntt::Dimension::THREE_D>;