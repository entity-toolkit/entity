#include "global.h"
#include "timer.h"
#include "simulation.h"
#include "sim_params.h"
#include "meshblock.h"
#include "input.h"

// #include "grid.h"
#include "cartesian.h"
// #include "spherical.h"
// #include "qspherical.h"

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
        m_meshblock {m_sim_params.m_extent, m_sim_params.m_resolution, m_sim_params.m_species} {
    initialize();
  }

  template <Dimension D>
  void Simulation<D>::initialize() {
    // pick the coordinate system
    if (m_sim_params.m_coord_system == "cartesian") {
      m_sim_params.m_min_cell_size
          = {(m_sim_params.m_extent[1] - m_sim_params.m_extent[0]) / (real_t)(m_sim_params.m_resolution[0])};
      m_meshblock.m_coord_system
          = std::make_unique<CartesianSystem<D>>(m_sim_params.m_resolution, m_sim_params.m_extent);
      // } else if (m_sim_params.m_coord_system == "spherical") {
      //   m_meshblock.m_coord_system = std::make_unique<SphericalSystem<D>>();
      // } else if (m_sim_params.m_coord_system == "qspherical") {
      //   auto r0 {m_sim_params.m_coord_parameters[0]};
      //   auto h {m_sim_params.m_coord_parameters[1]};
      //   m_meshblock.m_coord_system = std::make_unique<QSphericalSystem<D>>(r0, h);
    } else {
      throw std::logic_error("# coordinate system NOT IMPLEMENTED.");
    }

    // find timestep and effective cell size
    // if (m_sim_params.m_coord_system == "cartesian") {

    //   // if (m_dim == ONE_D) {
    //   //   real_t dx1 = m_meshblock.get_dx1();
    //   //   m_sim_params.m_min_cell_size = dx1;
    //   // } else if (m_dim == TWO_D) {
    //   //   real_t dx1 = m_meshblock.get_dx1();
    //   //   real_t dx2 = m_meshblock.get_dx2();
    //   //   m_sim_params.m_min_cell_size = 1.0 / std::sqrt(1.0 / (dx1 * dx1) + 1.0 / (dx2 * dx2));
    //   // } else if (m_dim == TWO_D) {
    //   //   real_t dx1 = m_meshblock.get_dx1();
    //   //   real_t dx2 = m_meshblock.get_dx2();
    //   //   real_t dx3 = m_meshblock.get_dx3();
    //   //   m_sim_params.m_min_cell_size = 1.0 / std::sqrt(1.0 / (dx1 * dx1) + 1.0 / (dx2 * dx2) + 1.0 / (dx3 * dx3));
    //   // }
    // // } else if ((m_sim_params.m_coord_system == "spherical") || (m_sim_params.m_coord_system == "qspherical")) {
    // //   if (m_dim == TWO_D) {
    // //     using index_t = NTTArray<real_t**>::size_type;
    // //     real_t min_dx {-1.0};
    // //     for (index_t i {0}; i < m_meshblock.m_resolution[0]; ++i) {
    // //       for (index_t j {0}; j < m_meshblock.m_resolution[1]; ++j) {
    // //         auto x1 = m_meshblock.convert_iTOx1(i);
    // //         auto x2 = m_meshblock.convert_jTOx2(j);
    // //         real_t dx1_ {m_meshblock.m_coord_system->hx1(x1, x2) * m_meshblock.get_dx1() * m_meshblock.get_dx1()};
    // //         real_t dx2_ {m_meshblock.m_coord_system->hx2(x1, x2) * m_meshblock.get_dx2() * m_meshblock.get_dx2()};
    // //         real_t dx = 1.0 / std::sqrt(1.0 / dx1_ + 1.0 / dx2_);
    // //         if ((min_dx >= dx) || (min_dx < 0.0)) { min_dx = dx; }
    // //       }
    // //     }
    // //     m_sim_params.m_min_cell_size = min_dx;
    // //   } else {
    // //     throw std::logic_error("# Error: CFL finding not implemented for 3D spherical.");
    // //   }
    // } else {
    //   throw std::logic_error("# Error: CFL finding not implemented for this coordinate system.");
    // }
    m_sim_params.m_timestep = m_sim_params.m_cfl * m_sim_params.m_min_cell_size;
  }

  template <Dimension D>
  void Simulation<D>::setIO(std::string_view infname, std::string_view outdirname) {
    m_sim_params.m_outputpath = outdirname;
    m_sim_params.m_inputfilename = infname;
  }

  template <Dimension D>
  void Simulation<D>::userInitialize() {
    m_pGen.userInitFields(m_sim_params, m_meshblock);
    fieldBoundaryConditions(0.0);
    m_pGen.userInitParticles(m_sim_params, m_meshblock);
    PLOGD << "Simulation initialized.";
  }

  template <Dimension D>
  void Simulation<D>::verify() {
    m_sim_params.verify();
    m_meshblock.verify(m_sim_params);
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
    PLOGI << "   coordinate system: " << (m_meshblock.m_coord_system->label);

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

    if (m_meshblock.particles.size() > 0) {
      PLOGI << "[particles]";
      for (std::size_t i {0}; i < m_meshblock.particles.size(); ++i) {
        PLOGI << "   [species #" << i + 1 << "]";
        PLOGI << "      label: " << m_meshblock.particles[i].get_label();
        PLOGI << "      mass: " << m_meshblock.particles[i].get_mass();
        PLOGI << "      charge: " << m_meshblock.particles[i].get_charge();
        PLOGI << "      pusher: " << stringifyParticlePusher(m_meshblock.particles[i].get_pusher());
        PLOGI << "      maxnpart: " << m_meshblock.particles[i].get_maxnpart() << " ("
              << m_meshblock.particles[i].get_npart() << ")";
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
