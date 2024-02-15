#include "grpic.h"

#include "wrapper.h"

#include "sim_params.h"

#include "utils/timer.h"

#include <plog/Log.h>

namespace ntt {

  template <Dimension D>
  void GRPIC<D>::Run() {
    auto  params = *(this->params());
    auto& mblock = this->meshblock;
    auto  timax  = (unsigned long)(params.totalRuntime() / mblock.timestep());

    ResetSimulation();
    this->PrintDetails();
    this->Verify();
    InitialStep();
    for (unsigned long ti { 0 }; ti < timax; ++ti) {
      PLOGV_(LogFile) << "step = " << this->m_tstep;
      PLOGV_(LogFile) << std::endl;
      StepForward();
    }
    WaitAndSynchronize();
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
    this->Communicate(Comm_D | Comm_D0 | Comm_B | Comm_B0);
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
    Faraday(HALF, gr_faraday::aux);
    /**
     * em0::B, em::B <- boundary conditions
     */
    this->Communicate(Comm_B | Comm_B0);
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
    this->Communicate(Comm_D | Comm_D0);
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
    this->Communicate(Comm_B | Comm_B0);
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
    this->Communicate(Comm_D | Comm_D0);
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
    this->Communicate(Comm_D | Comm_D0);
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
  }

  template <Dimension D>
  void GRPIC<D>::StepForward(const DiagFlags diag_flags) {
    NTTLog();
    const auto params     = *(this->params());
    const auto metadomain = *(this->metadomain());
    auto&      mblock     = this->meshblock;
    auto&      wrtr       = this->writer;
    auto&      pgen       = this->problem_generator;

    timer::Timers                   timers({ "FieldSolver",
                                             "FieldBoundaries",
                                             "CurrentDeposit",
                                             "ParticlePusher",
                                             "ParticleBoundaries",
                                             "UserSpecific" },
                         params.blockingTimers());
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
      this->Communicate(Comm_B | Comm_B0);
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

      /**
       * cur0::J <- current deposition
       *
       * Now: cur0::J at n+1/2
       */
      if (params.depositEnabled()) {
        timers.start("CurrentDeposit");
        CurrentsDeposit();

        timers.start("FieldBoundaries");
        this->CurrentsSynchronize();
        this->Communicate(Comm_J);
        CurrentsBoundaryConditions();
        timers.stop("FieldBoundaries");

        CurrentsFilter();
        timers.stop("CurrentDeposit");
      }

      timers.start("ParticleBoundaries");
      //this->ParticlesBoundaryConditions();
      this->Communicate(Comm_Prtl);
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
      this->Communicate(Comm_B | Comm_B0);
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

      if (params.depositEnabled()) {
        /**
         * em0::D <- (em0::D) <- cur::J
         *
         * Now: em0::D at n+1/2
         */
        AmpereCurrents(gr_ampere::aux);
      }

      timers.start("FieldBoundaries");
      /**
       * em0::D, em::D <- boundary conditions
       */
      this->Communicate(Comm_D | Comm_D0);
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

      if (params.depositEnabled()) {
        /**
         * em0::D <- (em0::D) <- cur0::J
         *
         * Now: em0::D at n+1
         */
        AmpereCurrents(gr_ampere::main);
      }

      /**
       * em::D <-> em0::D
       * em::B <-> em0::B
       * cur::J <-> cur0::J
       */
      SwapFields();
      timers.stop("FieldSolver");

      timers.start("FieldBoundaries");
      /**
       * em0::D, em::D <- boundary conditions
       */
      this->Communicate(Comm_D | Comm_D0);
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
  void GRPIC<D>::StepBackward() {}

} // namespace ntt

template class ntt::GRPIC<ntt::Dim2>;
template class ntt::GRPIC<ntt::Dim3>;
