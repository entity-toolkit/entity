#include "global.h"
#include "timer.h"
#include "grpic.h"
#include "sim_params.h"
#include "init_fields.hpp"

namespace ntt {

  template <Dimension D>
  void GRPIC<D>::mainloop() {
    unsigned long timax {static_cast<unsigned long>(this->m_sim_params.total_runtime() / this->m_mblock.timestep())};
    real_t time {0.0};
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
    /*
     * Initially: em0::B   --
     *            em0::D   --
     *            em::B    at -1/2
     *            em::D    at -1/2
     *
     *            cur0::J  --
     *            cur::J   --
     *
     *            aux::E   --
     *            aux::H   --
     *
     *            x_prtl   at -1/2
     *            u_prtl   at -1/2
     */

    /*
     * em0::D, em::D, em0::B, em::B <- boundary conditions
     */
    fieldBoundaryConditions(time, gr_bc::Dfield);
    fieldBoundaryConditions(time, gr_bc::Bfield);

    /*
     * em0::B <- em::B
     * em0::D <- em::D
     *
     * Now: em0::B & em0::D at -1/2
     */
    copyFieldsGR();

    /*
     * aux::E <- alpha * em::D + beta x em0::B
     * aux::H <- alpha * em::B0 - beta x em::D
     *
     * Now: aux::E & aux::H at -1/2
     */
    computeAuxESubstep(time, gr_getE::D0_B);
    computeAuxHSubstep(time, gr_getH::D_B0);
    /*
     * aux::E, aux::H <- boundary conditions
     */
    auxFieldBoundaryConditions(time, gr_bc::Efield);
    auxFieldBoundaryConditions(time, gr_bc::Hfield);

    /*
     * em0::B <- (em0::B) <- -curl aux::E
     *
     * Now: em0::B at 0
     */
    faradaySubstep(time, 0.5, gr_faraday::aux);
    /*
     * em0::B, em::B <- boundary conditions
     */
    fieldBoundaryConditions(time, gr_bc::Bfield);

    /*
     * em::D <- (em0::D) <- curl aux::H
     *
     * Now: em::D at 0
     */
    ampereSubstep(time, 0.5, gr_ampere::init);
    /*
     * em0::D, em::D <- boundary conditions
     */
    fieldBoundaryConditions(time, gr_bc::Dfield);

    /*
     * aux::E <- alpha * em::D + beta x em0::B
     * aux::H <- alpha * em0::B - beta x em::D
     *
     * Now: aux::E & aux::H at 0
     */
    computeAuxESubstep(time, gr_getE::D_B0);
    computeAuxHSubstep(time, gr_getH::D_B0);
    /*
     * aux::E, aux::H <- boundary conditions
     */
    auxFieldBoundaryConditions(time, gr_bc::Efield);
    auxFieldBoundaryConditions(time, gr_bc::Hfield);

    /*
     * em0::B <- (em::B) <- -curl aux::E
     *
     * Now: em0::B at 1/2
     */
    faradaySubstep(time, 1.0, gr_faraday::main);
    /*
     * em0::B, em::B <- boundary conditions
     */
    fieldBoundaryConditions(time, gr_bc::Bfield);

    /*
     * em0::D <- (em0::D) <- curl aux::H
     *
     * Now: em0::D at 1/2
     */
    ampereSubstep(time, 1.0, gr_ampere::aux);
    /*
     * em0::D, em::D <- boundary conditions
     */
    fieldBoundaryConditions(time, gr_bc::Dfield);

    /*
     * aux::H <- alpha * em0::B - beta x em0::D
     *
     * Now: aux::H at 1/2
     */
    computeAuxHSubstep(time, gr_getH::D0_B0);
    /*
     * aux::H <- boundary conditions
     */
    auxFieldBoundaryConditions(time, gr_bc::Hfield);

    /*
     * em0::D <- (em::D) <- curl aux::H
     *
     * Now: em0::D at 1
     *      em::D at 0
     */
    ampereSubstep(time, 1.0, gr_ampere::main);
    /*
     * em0::D, em::D <- boundary conditions
     */
    fieldBoundaryConditions(time, gr_bc::Dfield);

    /*
     * em::D <-> em0::D
     * em::B <-> em0::B
     * em::J <-> em0::J
     */
    swapFieldsGR();
    /*
     * Finally: em0::B   at -1/2
     *          em0::D   at 0
     *          em::B    at 1/2
     *          em::D    at 1
     *
     *          cur0::J  --
     *          cur::J   --
     *
     *          aux::E   --
     *          aux::H   --
     *
     *          x_prtl   at 1
     *          u_prtl   at 1/2
     */
  }

  template <Dimension D>
  void GRPIC<D>::step_forward(const real_t& time) {
    TimerCollection timers({"Field_solver", "Field_BC", "Curr_Deposit", "Prtl_Pusher"});
    /*
     * Initially: em0::B   at n-3/2
     *            em0::D   at n-1
     *            em::B    at n-1/2
     *            em::D    at n
     *
     *            cur0::J  --
     *            cur::J   at n-1/2
     *
     *            aux::E   --
     *            aux::H   --
     *
     *            x_prtl   at n
     *            u_prtl   at n-1/2
     */

    if (this->sim_params().enable_fieldsolver()) {
      timers.start(1);
      /*
       * em0::D <- (em0::D + em::D) / 2
       * em0::B <- (em0::B + em::B) / 2
       *
       * Now: em0::D at n-1/2
       *      em0::B at n-1
       */
      timeAverageDBSubstep(time);
      /*
       * aux::E <- alpha * em0::D + beta x em::B
       *
       * Now: aux::E at n-1/2
       */
      computeAuxESubstep(time, gr_getE::D0_B);
      /*
       * aux::E <- boundary conditions
       */
      auxFieldBoundaryConditions(time, gr_bc::Efield);
      /*
       * em0::B <- (em0::B) <- -curl aux::E
       *
       * Now: em0::B at n
       */
      faradaySubstep(time, 1.0, gr_faraday::aux);
      timers.stop(1);

      timers.start(2);
      /*
       * em0::B, em::B <- boundary conditions
       */
      fieldBoundaryConditions(time, gr_bc::Bfield);
      timers.stop(2);

      timers.start(1);
      /*
       * aux::H <- alpha * em0::B - beta x em::D
       *
       * Now: aux::H at n
       */
      computeAuxHSubstep(time, gr_getH::D_B0);
      /*
       * aux::H <- boundary conditions
       */
      auxFieldBoundaryConditions(time, gr_bc::Hfield);
      timers.stop(1);
    }

    // Push particles
    // x at n+1, u at n+1/2
    timers.start(4);
    timers.stop(4);

    /*
     * cur0::J <- current deposition
     *
     * Now: cur0::J at n+1/2
     */
    timers.start(3);
    timers.stop(3);

    if (this->sim_params().enable_fieldsolver()) {
      timers.start(1);
      /*
       * cur::J <- (cur0::J + cur::J) / 2
       *
       * Now: cur::J at n
       */
      timeAverageJSubstep(time);
      /*
       * aux::Е <- alpha * em::D + beta x em0::B
       *
       * Now: aux::Е at n
       */
      computeAuxESubstep(time, gr_getE::D_B0);
      /*
       * aux::Е <- boundary conditions
       */
      auxFieldBoundaryConditions(time, gr_bc::Efield);
      /*
       * em0::B <- (em::B) <- -curl aux::E
       *
       * Now: em0::B at n+1/2
       *      em::B at n-1/2
       */
      faradaySubstep(time, 1.0, gr_faraday::main);
      timers.stop(1);

      timers.start(2);
      /*
       * em0::B, em::B <- boundary conditions
       */
      fieldBoundaryConditions(time, gr_bc::Bfield);
      timers.stop(2);

      timers.start(1);
      /*
       * em0::D <- (em0::D) <- curl aux::H
       *
       * Now: em0::D at n+1/2
       */
      ampereSubstep(time, 1.0, gr_ampere::aux);
      timers.stop(1);

      timers.start(2);
      /*
       * em0::D, em::D <- boundary conditions
       */
      fieldBoundaryConditions(time, gr_bc::Dfield);
      timers.stop(2);

      timers.start(1);
      /*
       * aux::H <- alpha * em0::B - beta x em0::D
       *
       * Now: aux::H at n+1/2
       */
      computeAuxHSubstep(time, gr_getH::D0_B0);
      /*
       * aux::H <- boundary conditions
       */
      auxFieldBoundaryConditions(time, gr_bc::Hfield);
      /*
       * em0::D <- (em::D) <- curl aux::H
       *
       * Now: em0::D at n+1
       *      em::D at n
       */
      ampereSubstep(time, 1.0, gr_ampere::main);

      /*
       * em::D <-> em0::D
       * em::B <-> em0::B
       * em::J <-> em0::J
       */
      swapFieldsGR();
      timers.stop(1);

      timers.start(2);
      /*
       * em0::D, em::D <- boundary conditions
       */
      fieldBoundaryConditions(time, gr_bc::Dfield);
      timers.stop(2);
    }

    /*
     * Finally: em0::B   at n-1/2
     *          em0::D   at n
     *          em::B    at n+1/2
     *          em::D    at n+1
     *
     *          cur0::J  (at n)
     *          cur::J   at n+1/2
     *
     *          aux::E   (at n+1/2)
     *          aux::H   (at n)
     *
     *          x_prtl   at n+1
     *          u_prtl   at n+1/2
     */
    timers.printAll(millisecond);
    computeVectorPotential();
  }

  template <>
  void GRPIC<Dimension::TWO_D>::computeVectorPotential() {
    Kokkos::parallel_for("computeVectorPotential",
                         (this->m_mblock).loopActiveCells(),
                         Compute_Aphi<Dimension::TWO_D>(this->m_mblock, (real_t)(1.0)));
  }

  template <>
  void GRPIC<Dimension::THREE_D>::computeVectorPotential() {}

} // namespace ntt

template class ntt::GRPIC<ntt::Dimension::TWO_D>;
template class ntt::GRPIC<ntt::Dimension::THREE_D>;

// template <Dimension D>
// void GRPIC<D>::step_backward(const real_t& time) {
//   TimerCollection timers({"Field_solver", "Field_BC", "Curr_Deposit", "Prtl_Pusher"});

//   // Initially: B0 at n-3/2, B at n-1/2, D0 at n-1, D at n, x at n, u at n-1/2, J0 at n-1, J at n-1/2

//   timers.start(1);
//   // B0 at n-1, B at n-1/2, D0 at n-1/2, D at n
//   timeAverageDBSubstep(time);
//   // E at n-1/2 with B and D0
//   computeAuxE_D_B0Substep(time, 0);
//   auxFieldBoundaryConditions(time, 0);
//   // B0 at n, B at n-1/2
//   faradaySubstep(time, -1.0, 0);
//   timers.stop(1);

//   timers.start(2);
//   fieldBoundaryConditions(time, 1);
//   timers.stop(2);

//   timers.start(1);
//   // H at n with B0 and D
//   computeAuxHSubstep(time, 0);
//   auxFieldBoundaryConditions(time, 1);
//   timers.stop(1);

//   // Push particles
//   // x at n+1, u at n+1/2
//   timers.start(4);
//   timers.stop(4);

//   // Current deposition
//   // J0 at n+1/2, J at n-1/2
//   timers.start(3);
//   timers.stop(3);

//   timers.start(1);
//   // J0 at n+1/2, J at n
//   timeAverageJSubstep(time);
//   // E at n with B0 and D
//   computeAuxE_D_B0Substep(time, 1);
//   auxFieldBoundaryConditions(time, 0);
//   // B0 at n+1/2, B at n-1/2
//   faradaySubstep(time, -1.0, 1);
//   timers.stop(1);

//   timers.start(2);
//   fieldBoundaryConditions(time, 1);
//   timers.stop(2);

//   timers.start(1);
//   // D0 at n+1/2, D at n
//   ampereSubstep(time, -1.0, 0);
//   timers.stop(1);

//   timers.start(2);
//   fieldBoundaryConditions(time, 0);
//   timers.stop(2);

//   timers.start(1);
//   // H at n+1/2 with B0 and D0
//   computeAuxHSubstep(time, 1);
//   auxFieldBoundaryConditions(time, 1);
//   // D0 at n+1, D at n
//   ampereSubstep(time, -1.0, 1);

//   // Final: B0 at n-1/2, B at n+1/2, D0 at n, D at n+1, x at n+1, u at n+1/2, J0 at n, J at n+1/2
//   swap_em_cur(this->m_mblock);
//   timers.stop(1);

//   timers.start(2);
//   fieldBoundaryConditions(time, 0);
//   timers.stop(2);

//   timers.printAll(millisecond);
// }