#include "global.h"
#include "timer.h"
#include "simulation.h"
#include "sim_params.h"
// #include "meshblock.h"
#include "input.h"

// #include "cartesian.h"
// #include "spherical.h"
// #include "qspherical.h"

#include <plog/Log.h>
#include <toml/toml.hpp>

#include <memory>
#include <cmath>
#include <stdexcept>

namespace ntt {

  template <Dimension D, SimulationType S>
  Simulation<D, S>::Simulation(const toml::value& inputdata)
    : m_sim_params {inputdata, D}, m_pGen {m_sim_params}, m_mblock {m_sim_params.resolution(), m_sim_params.species()} {
    initialize();
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::initialize() {
    // // pick the coordinate system
    // if (m_sim_params.coord_system() == "cartesian") {
    //   m_mblock.grid = std::make_unique<CartesianSystem<D>>(m_sim_params.resolution(), m_sim_params.extent());
    // } else if (m_sim_params.coord_system() == "spherical") {
    //   m_mblock.grid = std::make_unique<SphericalSystem<D>>(m_sim_params.resolution(), m_sim_params.extent());
    // } else if (m_sim_params.coord_system() == "qspherical") {
    //   auto r0 {m_sim_params.coord_parameters(0)};
    //   auto h {m_sim_params.coord_parameters(1)};
    //   m_mblock.grid = std::make_unique<QSphericalSystem<D>>(m_sim_params.resolution(), m_sim_params.extent(), r0, h);
    // } else {
    //   throw std::logic_error("# coordinate system NOT IMPLEMENTED.");
    // }

    // find timestep and effective cell size
    m_mblock.set_min_cell_size(1.0);
    m_mblock.set_timestep(1.0);
  }

  // template <Dimension D>
  // void Simulation<D>::setIO(std::string_view infname, std::string_view outdirname) {
  //   m_sim_params.m_outputpath = outdirname;
  //   m_sim_params.m_inputfilename = infname;
  // }

  // template <Dimension D>
  // void Simulation<D>::userInitialize() {
  //   m_pGen.userInitFields(m_sim_params, mblock);
  //   fieldBoundaryConditions(0.0);
  //   m_pGen.userInitParticles(m_sim_params, mblock);
  //   PLOGD << "Simulation initialized.";
  // }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::verify() {
    // m_sim_params.verify();
    // mblock.verify(m_sim_params);
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::printDetails() {
    PLOGI << "[Simulation details]";
    PLOGI << "   title: " << m_sim_params.title();
    PLOGI << "   type: " << stringifySimulationType(S);
    PLOGI << "   total runtime: " << m_sim_params.total_runtime();
    PLOGI << "   dt: " << m_mblock.timestep() << " ["
          << static_cast<int>(m_sim_params.total_runtime() / m_mblock.timestep()) << " steps]";

    PLOGI << "[domain]";
    PLOGI << "   dimension: " << static_cast<short>(D) << "D";
    // PLOGI << "   coordinate system: " << (mblock.grid->label);

    std::string bc {"   boundary conditions: { "};
    for (auto& b : m_sim_params.boundaries()) {
      bc += stringifyBoundaryCondition(b) + " x ";
    }
    bc.erase(bc.size() - 3);
    bc += " }";
    PLOGI << bc;

    std::string res {"   resolution: { "};
    for (auto& r : m_sim_params.resolution()) {
      res += std::to_string(r) + " x ";
    }
    res.erase(res.size() - 3);
    res += " }";
    PLOGI << res;

    std::string ext {"   extent: "};
    for (std::size_t i {0}; i < m_sim_params.extent().size(); i += 2) {
      ext
        += "{" + std::to_string(m_sim_params.extent()[i]) + ", " + std::to_string(m_sim_params.extent()[i + 1]) + "} ";
    }
    PLOGI << ext;

    std::string cell {"   cell size: "};
    cell += std::to_string(m_mblock.min_cell_size());
    PLOGI << cell;

    PLOGI << "[fiducial parameters]";
    PLOGI << "   ppc0: " << m_sim_params.ppc0();
    PLOGI << "   rho0: " << m_sim_params.larmor0() << " ["
          << m_sim_params.larmor0() / m_mblock.min_cell_size() << " cells]";
    PLOGI << "   c_omp0: " << m_sim_params.skindepth0() << " ["
          << m_sim_params.skindepth0() / m_mblock.min_cell_size() << " cells]";
    PLOGI << "   sigma0: " << m_sim_params.sigma0();
    PLOGI << "   q0: " << m_sim_params.charge0();
    PLOGI << "   B0: " << m_sim_params.B0();

    if (m_mblock.particles.size() > 0) {
      PLOGI << "[particles]";
      int i {0};
      for (auto& prtls : m_mblock.particles) {
        PLOGI << "   [species #" << i + 1 << "]";
        PLOGI << "      label: " << prtls.label();
        PLOGI << "      mass: " << prtls.mass();
        PLOGI << "      charge: " << prtls.charge();
        PLOGI << "      pusher: " << stringifyParticlePusher(prtls.pusher());
        PLOGI << "      maxnpart: " << prtls.maxnpart() << " (" << prtls.npart() << ")";
        ++i;
      }
    } else {
      PLOGI << "[no particles]";
    }
  }

  // template <Dimension D>
  // void Simulation<D>::finalize() {
  //   PLOGD << "Simulation finalized.";
  // }

  // template <Dimension D>
  // void Simulation<D>::step_forward(const real_t& time) {
  //   TimerCollection timers({"Field_Solver", "Field_BC", "Curr_Deposit", "Prtl_Pusher"});
  //   {
  //     timers.start(1);
  //     faradaySubstep(time, 0.5);
  //     timers.stop(1);
  //   }

  //   {
  //     timers.start(2);
  //     fieldBoundaryConditions(time);
  //     timers.stop(2);
  //   }

  //   {
  //     timers.start(4);
  //     pushParticlesSubstep(time);
  //     timers.stop(4);
  //   }

  //   // depositSubstep(time);

  //   {
  //     timers.start(2);
  //     particleBoundaryConditions(time);
  //     timers.stop(2);
  //   }
  //   // BC currents

  //   {
  //     timers.start(1);
  //     faradaySubstep(time, 0.5);
  //     timers.stop(1);
  //   }

  //   {
  //     timers.start(2);
  //     fieldBoundaryConditions(time);
  //     timers.stop(2);
  //   }

  //   {
  //     timers.start(1);
  //     ampereSubstep(time, 1.0);
  //     addCurrentsSubstep(time);
  //     resetCurrentsSubstep(time);
  //     timers.stop(1);
  //   }

  //   {
  //     timers.start(2);
  //     fieldBoundaryConditions(time);
  //     timers.stop(2);
  //   }
  //   timers.printAll(millisecond);
  // }

  // template <Dimension D>
  // void Simulation<D>::step_backward(const real_t& time) {
  //   TimerCollection timers({"Field_Solver", "Field_BC", "Curr_Deposit", "Prtl_Pusher"});
  //   {
  //     timers.start(1);
  //     ampereSubstep(time, -1.0);
  //     timers.stop(1);
  //   }

  //   {
  //     timers.start(2);
  //     fieldBoundaryConditions(time);
  //     timers.stop(2);
  //   }

  //   {
  //     timers.start(1);
  //     faradaySubstep(time, -1.0);
  //     timers.stop(1);
  //   }

  //   {
  //     timers.start(2);
  //     fieldBoundaryConditions(time);
  //     timers.stop(2);
  //   }
  //   timers.printAll(millisecond);
  // }

  // template <Dimension D>
  // void Simulation<D>::mainloop() {
  //   PLOGD << "Simulation mainloop started.";

  //   unsigned long timax {
  //       static_cast<unsigned long>(m_sim_params.m_runtime / m_sim_params.m_timestep)};
  //   real_t time {0.0};
  //   for (unsigned long ti {0}; ti < timax; ++ti) {
  //     PLOGD << "t = " << time;
  //     step_forward(time);
  //     time += m_sim_params.m_timestep;
  //   }
  //   PLOGD << "Simulation mainloop finished.";
  // }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::run() {
    // setIO(infname, outdirname);
    // userInitialize();
    verify();
    PLOGD << "Prerun check passed";
    printDetails();
    PLOGD << "Simulation details printed";
    // mainloop();
    // finalize();
  }

} // namespace ntt

template class ntt::Simulation<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
template class ntt::Simulation<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
template class ntt::Simulation<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;
