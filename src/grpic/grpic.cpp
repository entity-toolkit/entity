#include "grpic.h"

#include "wrapper.h"

#include "sim_params.h"
#include "timer.h"

#include <plog/Log.h>

namespace ntt {

  template <Dimension D>
  void GRPIC<D>::Run() {
    // register the content of em fields
    Simulation<D, GRPICEngine>::Initialize();
    Simulation<D, GRPICEngine>::Verify();
    {
      auto  params = *(this->params());
      auto& mblock = this->meshblock;
      auto  timax  = (unsigned long)(params.totalRuntime() / mblock.timestep());

      ResetSimulation();
      Simulation<D, GRPICEngine>::PrintDetails();
      InitialStep();
      for (unsigned long ti { 0 }; ti < timax; ++ti) {
        PLOGV_(LogFile) << "step = " << this->m_tstep;
        PLOGV_(LogFile) << std::endl;
        StepForward();
      }
      WaitAndSynchronize();
    }
    Simulation<D, GRPICEngine>::Finalize();
  }

  template <Dimension D>
  void GRPIC<D>::InitializeSetup() {
    auto  params = *(this->params());
    auto& mblock = this->meshblock;
    problem_generator.UserInitFields(params, mblock);
    problem_generator.UserInitParticles(params, mblock);
  }

  template <Dimension D>
  void GRPIC<D>::InitialStep() {
    /**
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

    /**
     * em0::D, em::D, em0::B, em::B <- boundary conditions
     */
    FieldsBoundaryConditions(gr_bc::Dfield);
    FieldsBoundaryConditions(gr_bc::Bfield);

    /**
     * em0::B <- em::B
     * em0::D <- em::D
     *
     * Now: em0::B & em0::D at -1/2
     */
    CopyFields();

    /**
     * aux::E <- alpha * em::D + beta x em0::B
     * aux::H <- alpha * em::B0 - beta x em::D
     *
     * Now: aux::E & aux::H at -1/2
     */
    ComputeAuxE(gr_getE::D0_B);
    ComputeAuxH(gr_getH::D_B0);
    /**
     * aux::E, aux::H <- boundary conditions
     */
    AuxFieldsBoundaryConditions(gr_bc::Efield);
    AuxFieldsBoundaryConditions(gr_bc::Hfield);

    /**
     * em0::B <- (em0::B) <- -curl aux::E
     *
     * Now: em0::B at 0
     */
    Faraday(0.5, gr_faraday::aux);
    /**
     * em0::B, em::B <- boundary conditions
     */
    FieldsBoundaryConditions(gr_bc::Bfield);

    /**
     * em::D <- (em0::D) <- curl aux::H
     *
     * Now: em::D at 0
     */
    Ampere(HALF, gr_ampere::init);
    /**
     * em0::D, em::D <- boundary conditions
     */
    FieldsBoundaryConditions(gr_bc::Dfield);

    /**
     * aux::E <- alpha * em::D + beta x em0::B
     * aux::H <- alpha * em0::B - beta x em::D
     *
     * Now: aux::E & aux::H at 0
     */
    ComputeAuxE(gr_getE::D_B0);
    ComputeAuxH(gr_getH::D_B0);
    /**
     * aux::E, aux::H <- boundary conditions
     */
    AuxFieldsBoundaryConditions(gr_bc::Efield);
    AuxFieldsBoundaryConditions(gr_bc::Hfield);

    // !ADD: GR -- particles?

    /**
     * em0::B <- (em::B) <- -curl aux::E
     *
     * Now: em0::B at 1/2
     */
    Faraday(ONE, gr_faraday::main);
    /**
     * em0::B, em::B <- boundary conditions
     */
    FieldsBoundaryConditions(gr_bc::Bfield);

    /**
     * em0::D <- (em0::D) <- curl aux::H
     *
     * Now: em0::D at 1/2
     */
    Ampere(ONE, gr_ampere::aux);
    /**
     * em0::D, em::D <- boundary conditions
     */
    FieldsBoundaryConditions(gr_bc::Dfield);

    /**
     * aux::H <- alpha * em0::B - beta x em0::D
     *
     * Now: aux::H at 1/2
     */
    ComputeAuxH(gr_getH::D0_B0);
    /**
     * aux::H <- boundary conditions
     */
    AuxFieldsBoundaryConditions(gr_bc::Hfield);

    /**
     * em0::D <- (em::D) <- curl aux::H
     *
     * Now: em0::D at 1
     *      em::D at 0
     */
    Ampere(ONE, gr_ampere::main);
    /**
     * em0::D, em::D <- boundary conditions
     */
    FieldsBoundaryConditions(gr_bc::Dfield);

    /**
     * em::D <-> em0::D
     * em::B <-> em0::B
     * em::J <-> em0::J
     */
    SwapFields();
    /**
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
    auto& mblock = this->meshblock;
  }

  template <Dimension D>
  void GRPIC<D>::StepForward() {
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
                                             "UserSpecific" },
                         params.blockingTimers());
    static std::vector<double>      dead_fractions  = {};
    static std::vector<long double> tstep_durations = {};
    /**
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

    if (params.fieldsolverEnabled()) {
      timers.start("FieldSolver");
      /**
       * em0::D <- (em0::D + em::D) / 2
       * em0::B <- (em0::B + em::B) / 2
       *
       * Now: em0::D at n-1/2
       *      em0::B at n-1
       */
      TimeAverageDB();
      /**
       * aux::E <- alpha * em0::D + beta x em::B
       *
       * Now: aux::E at n-1/2
       */
      ComputeAuxE(gr_getE::D0_B);
      /**
       * aux::E <- boundary conditions
       */
      AuxFieldsBoundaryConditions(gr_bc::Efield);
      /**
       * em0::B <- (em0::B) <- -curl aux::E
       *
       * Now: em0::B at n
       */
      Faraday(ONE, gr_faraday::aux);
      timers.stop("FieldSolver");

      timers.start("FieldBoundaries");
      /**
       * em0::B, em::B <- boundary conditions
       */
      FieldsBoundaryConditions(gr_bc::Bfield);
      timers.stop("FieldBoundaries");

      timers.start("FieldSolver");
      /**
       * aux::H <- alpha * em0::B - beta x em::D
       *
       * Now: aux::H at n
       */
      ComputeAuxH(gr_getH::D_B0);
      /**
       * aux::H <- boundary conditions
       */
      AuxFieldsBoundaryConditions(gr_bc::Hfield);
      timers.stop("FieldSolver");
    }

    {
      /**
       * x_prtl, u_prtl <- em::D, em0::B
       *
       * Now: x_prtl at n + 1, u_prtl at n + 1/2
       */
      timers.start("ParticlePusher");
      ParticlesPush();
      timers.stop("ParticlePusher");

      timers.start("UserSpecific");
      pgen.UserDriveParticles(this->m_time, params, mblock);
      timers.stop("UserSpecific");

      timers.start("ParticleBoundaries");
      ParticlesBoundaryConditions();
      timers.stop("ParticleBoundaries");

      /**
       * cur0::J <- current deposition
       *
       * Now: cur0::J at n+1/2
       */
      if (params.depositEnabled()) {
        timers.start("CurrentDeposit");
        // !ADD: GR -- reset + deposit

        timers.start("FieldBoundaries");
        // !ADD: GR -- synchronize + exchange + bc
        timers.stop("FieldBoundaries");

        // !ADD: GR -- filter
        timers.stop("CurrentDeposit");
      }

      timers.start("ParticleBoundaries");
      // !ADD: GR -- particle exchange ?
      if ((params.shuffleInterval() > 0) && (this->m_tstep % params.shuffleInterval() == 0)) {
        dead_fractions = mblock.RemoveDeadParticles(params.maxDeadFraction());
      }
      timers.stop("ParticleBoundaries");
    }

    if (params.fieldsolverEnabled()) {
      timers.start("FieldSolver");
      /**
       * cur::J <- (cur0::J + cur::J) / 2
       *
       * Now: cur::J at n
       */
      TimeAverageJ();
      /**
       * aux::Е <- alpha * em::D + beta x em0::B
       *
       * Now: aux::Е at n
       */
      ComputeAuxE(gr_getE::D_B0);
      /**
       * aux::Е <- boundary conditions
       */
      AuxFieldsBoundaryConditions(gr_bc::Efield);
      /**
       * em0::B <- (em::B) <- -curl aux::E
       *
       * Now: em0::B at n+1/2
       *      em::B at n-1/2
       */
      Faraday(ONE, gr_faraday::main);
      timers.stop("FieldSolver");

      timers.start("FieldBoundaries");
      /**
       * em0::B, em::B <- boundary conditions
       */
      FieldsBoundaryConditions(gr_bc::Bfield);
      timers.stop("FieldBoundaries");

      timers.start("FieldSolver");
      /**
       * em0::D <- (em0::D) <- curl aux::H
       *
       * Now: em0::D at n+1/2
       */
      Ampere(ONE, gr_ampere::aux);
      timers.stop("FieldSolver");

      timers.start("FieldBoundaries");
      /**
       * em0::D, em::D <- boundary conditions
       */
      FieldsBoundaryConditions(gr_bc::Dfield);
      timers.stop("FieldBoundaries");

      timers.start("FieldSolver");
      /**
       * aux::H <- alpha * em0::B - beta x em0::D
       *
       * Now: aux::H at n+1/2
       */
      ComputeAuxH(gr_getH::D0_B0);
      /**
       * aux::H <- boundary conditions
       */
      AuxFieldsBoundaryConditions(gr_bc::Hfield);
      /**
       * em0::D <- (em::D) <- curl aux::H
       *
       * Now: em0::D at n+1
       *      em::D at n
       */
      Ampere(ONE, gr_ampere::main);

      /**
       * em::D <-> em0::D
       * em::B <-> em0::B
       * em::J <-> em0::J
       */
      SwapFields();
      timers.stop("FieldSolver");

      timers.start("FieldBoundaries");
      /**
       * em0::D, em::D <- boundary conditions
       */
      FieldsBoundaryConditions(gr_bc::Dfield);
      timers.stop("FieldBoundaries");
    }

    /**
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
    timers.start("Output");
    wrtr.WriteAll(params, mblock, this->m_time, this->m_tstep);
    timers.stop("Output");

    this->PrintDiagnostics(
      this->m_tstep, this->m_time, dead_fractions, timers, tstep_durations);

    this->m_time += mblock.timestep();
    this->m_tstep++;
  }

  template <Dimension D>
  void GRPIC<D>::StepBackward() {}

  // template <>
  // void GRPIC<Dim2>::computeVectorPotential() {
  //   Kokkos::parallel_for("computeVectorPotential",
  //                        (this->m_mblock).rangeActiveCells(),
  //                        Compute_Aphi<Dim2>(this->m_mblock, (real_t)(1.0)));
  // }

  // template <>
  // void GRPIC<Dim3>::computeVectorPotential() {}

  // template <>
  // Inline void Compute_Aphi<Dim2>::operator()(index_t i, index_t j) const {
  //   real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
  //   for (int k = (int)(i2_min - N_GHOSTS) + 1; k <= (int)(j - N_GHOSTS); ++k) {
  //     real_t sqrt_detH_ij1 { m_mblock.metric.sqrt_det_h({ i_, (real_t)k - HALF }) };
  //     real_t sqrt_detH_ij2 { m_mblock.metric.sqrt_det_h({ i_, (real_t)k + HALF }) };
  //     int    k1 { k + N_GHOSTS };
  //     m_mblock.aphi(i, j, 0) += HALF
  //                               * (sqrt_detH_ij1 * m_mblock.em(i, k1 - 1, em::bx1)
  //                                  + sqrt_detH_ij2 * m_mblock.em(i, k1, em::bx1));
  //   }
  // }

  // template <>
  // Inline void Compute_Aphi<Dim3>::operator()(index_t, index_t) const {
  //   // 3D GRPIC not implemented
  // }

}    // namespace ntt

template class ntt::GRPIC<ntt::Dim2>;
template class ntt::GRPIC<ntt::Dim3>;

// template <Dimension D>
// void GRPIC<D>::step_backward(const real_t& time) {
//   TimerCollection timers({"Field_solver", "Field_BC", "Curr_Deposit", "Prtl_Pusher"});

//   // Initially: B0 at n-3/2, B at n-1/2, D0 at n-1, D at n, x at n, u at n-1/2, J0 at n-1, J
//   at n-1/2

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

//   // Final: B0 at n-1/2, B at n+1/2, D0 at n, D at n+1, x at n+1, u at n+1/2, J0 at n, J at
//   n+1/2 swap_em_cur(this->m_mblock); timers.stop(1);

//   timers.start(2);
//   fieldBoundaryConditions(time, 0);
//   timers.stop(2);

//   timers.printAll(millisecond);
// }
