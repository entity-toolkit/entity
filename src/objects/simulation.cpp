#include "global.h"
#include "timer.h"
#include "simulation.h"
#include "sim_params.h"
#include "meshblock.h"
#include "input.h"

#include "cartesian.h"
#include "spherical.h"
#include "qspherical.h"

#include <plog/Log.h>
#include <toml/toml.hpp>

#include <memory>
#include <cmath>
#include <stdexcept>

namespace ntt {

  template <Dimension D>
  Simulation<D>::Simulation(const toml::value& inputdata)
      : m_sim_params {inputdata, m_dim},
        m_pGen {m_sim_params},
        mblock {m_sim_params.m_resolution, m_sim_params.m_species} {
    initialize();
  }

  template <Dimension D>
  void Simulation<D>::initialize() {
    // pick the coordinate system
    if (m_sim_params.m_coord_system == "cartesian") {
      mblock.grid
        = std::make_unique<CartesianSystem<D>>(m_sim_params.m_resolution, m_sim_params.m_extent);
    } else if (m_sim_params.m_coord_system == "spherical") {
      mblock.grid
        = std::make_unique<SphericalSystem<D>>(m_sim_params.m_resolution, m_sim_params.m_extent);
    } else if (m_sim_params.m_coord_system == "qspherical") {
      auto r0 {m_sim_params.m_coord_parameters[0]};
      auto h {m_sim_params.m_coord_parameters[1]};
      mblock.grid = std::make_unique<QSphericalSystem<D>>(m_sim_params.m_resolution, m_sim_params.m_extent, r0, h);
    } else {
      throw std::logic_error("# coordinate system NOT IMPLEMENTED.");
    }

    // find timestep and effective cell size
    m_sim_params.m_min_cell_size = mblock.grid->findSmallestCell();
    m_sim_params.m_timestep = m_sim_params.m_cfl * m_sim_params.m_min_cell_size;
  }

  template <Dimension D>
  void Simulation<D>::setIO(std::string_view infname, std::string_view outdirname) {
    m_sim_params.m_outputpath = outdirname;
    m_sim_params.m_inputfilename = infname;
  }

  template <Dimension D>
  void Simulation<D>::userInitialize() {
    m_pGen.userInitFields(m_sim_params, mblock);
    fieldBoundaryConditions(0.0);
    m_pGen.userInitParticles(m_sim_params, mblock);
    PLOGD << "Simulation initialized.";
  }

  template <Dimension D>
  void Simulation<D>::verify() {
    m_sim_params.verify();
    mblock.verify(m_sim_params);
    PLOGD << "Simulation prerun check passed.";
  }

  template <Dimension D>
  void Simulation<D>::printDetails() {
    PLOGI << "[Simulation details]";
    PLOGI << "   title: " << m_sim_params.m_title;
    PLOGI << "   type: " << stringifySimulationType(m_sim_params.m_simtype);
    PLOGI << "   total runtime: " << m_sim_params.m_runtime;
    PLOGI << "   dt: " << m_sim_params.m_timestep << " ["
          << static_cast<int>(m_sim_params.m_runtime / m_sim_params.m_timestep) << " steps]";

    PLOGI << "[domain]";
    PLOGI << "   dimension: " << m_dim << "D";
    PLOGI << "   coordinate system: " << (mblock.grid->label);

    std::string bc {"   boundary conditions: { "};
    for (auto& b : m_sim_params.m_boundaries) {
      bc += stringifyBoundaryCondition(b) + " x ";
    }
    bc.erase(bc.size() - 3);
    bc += " }";
    PLOGI << bc;

    std::string res {"   resolution: { "};
    for (auto& r : m_sim_params.m_resolution) {
      res += std::to_string(r) + " x ";
    }
    res.erase(res.size() - 3);
    res += " }";
    PLOGI << res;

    std::string ext {"   extent: "};
    for (std::size_t i {0}; i < m_sim_params.m_extent.size(); i += 2) {
      ext += "{" + std::to_string(m_sim_params.m_extent[i]) + ", "
             + std::to_string(m_sim_params.m_extent[i + 1]) + "} ";
    }
    PLOGI << ext;

    std::string cell {"   cell size: "};
    cell += std::to_string(m_sim_params.m_min_cell_size);
    PLOGI << cell;

    PLOGI << "[fiducial parameters]";
    PLOGI << "   ppc0: " << m_sim_params.m_ppc0;
    PLOGI << "   rho0: " << m_sim_params.m_larmor0 << " ["
          << m_sim_params.m_larmor0 / m_sim_params.m_min_cell_size << " d_min]";
    PLOGI << "   c_omp0: " << m_sim_params.m_skindepth0 << " ["
          << m_sim_params.m_skindepth0 / m_sim_params.m_min_cell_size << " d_min]";
    PLOGI << "   sigma0: " << m_sim_params.m_sigma0;
    PLOGI << "   q0: " << m_sim_params.m_charge0;
    PLOGI << "   B0: " << m_sim_params.m_B0;

    if (mblock.particles.size() > 0) {
      PLOGI << "[particles]";
      for (std::size_t i {0}; i < mblock.particles.size(); ++i) {
        PLOGI << "   [species #" << i + 1 << "]";
        PLOGI << "      label: " << mblock.particles[i].get_label();
        PLOGI << "      mass: " << mblock.particles[i].get_mass();
        PLOGI << "      charge: " << mblock.particles[i].get_charge();
        PLOGI << "      pusher: " << stringifyParticlePusher(mblock.particles[i].get_pusher());
        PLOGI << "      maxnpart: " << mblock.particles[i].get_maxnpart() << " ("
              << mblock.particles[i].get_npart() << ")";
      }
    } else {
      PLOGI << "[no particles]";
    }
  }

  template <Dimension D>
  void Simulation<D>::finalize() {
    PLOGD << "Simulation finalized.";
  }

  template <Dimension D>
  void Simulation<D>::step_forward(const real_t& time) {
    TimerCollection timers({"Field_Solver", "Field_BC", "Curr_Deposit", "Prtl_Pusher"});
    {
      timers.start(1);
      faradaySubstep(time, 0.5);
      timers.stop(1);
    }

    {
      timers.start(2);
      fieldBoundaryConditions(time);
      timers.stop(2);
    }

    {
      timers.start(4);
      pushParticlesSubstep(time);
      timers.stop(4);
    }

    // depositSubstep(time);

    {
      timers.start(2);
      particleBoundaryConditions(time);
      timers.stop(2);
    }
    // BC currents

    {
      timers.start(1);
      faradaySubstep(time, 0.5);
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
      addCurrentsSubstep(time);
      resetCurrentsSubstep(time);
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
  void Simulation<D>::step_backward(const real_t& time) {
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

  template <Dimension D>
  void Simulation<D>::mainloop() {
    PLOGD << "Simulation mainloop started.";

    unsigned long timax {
        static_cast<unsigned long>(m_sim_params.m_runtime / m_sim_params.m_timestep)};
    real_t time {0.0};
    for (unsigned long ti {0}; ti < timax; ++ti) {
      PLOGD << "t = " << time;
      step_forward(time);
      time += m_sim_params.m_timestep;
    }
    PLOGD << "Simulation mainloop finished.";
  }

  template <Dimension D>
  void Simulation<D>::run(std::string_view infname, std::string_view outdirname) {
    setIO(infname, outdirname);
    userInitialize();
    verify();
    printDetails();
    mainloop();
    finalize();
  }

} // namespace ntt

template class ntt::Simulation<ntt::ONE_D>;
template class ntt::Simulation<ntt::TWO_D>;
template class ntt::Simulation<ntt::THREE_D>;
