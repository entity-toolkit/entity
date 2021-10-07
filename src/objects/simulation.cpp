#include "global.h"
#include "timer.h"
#include "simulation.h"
#include "sim_params.h"
#include "meshblock.h"

#include <plog/Log.h>
#include <toml/toml.hpp>

#include <stdexcept>

namespace ntt {

template <Dimension D>
Simulation<D>::Simulation(const toml::value& inputdata)
    : m_sim_params{inputdata, m_dim}
      // m_pGen{m_sim_params}
      {}

Simulation1D::Simulation1D(const toml::value& inputdata)
    : Simulation<ONE_D>{inputdata},
      m_meshblock{m_sim_params.m_resolution, m_sim_params.m_species} {
  // TODO: meshblock extent can be different from global one
  m_meshblock.set_extent(m_sim_params.m_extent);
  m_meshblock.set_coord_system(m_sim_params.m_coord_system);
}

Simulation2D::Simulation2D(const toml::value& inputdata)
    : Simulation<TWO_D>{inputdata},
      m_meshblock{m_sim_params.m_resolution, m_sim_params.m_species} {
  // TODO: meshblock extent can be different from global one
  m_meshblock.set_extent(m_sim_params.m_extent);
  m_meshblock.set_coord_system(m_sim_params.m_coord_system);
}

Simulation3D::Simulation3D(const toml::value& inputdata)
    : Simulation<THREE_D>{inputdata},
      m_meshblock{m_sim_params.m_resolution, m_sim_params.m_species} {
  // TODO: meshblock extent can be different from global one
  m_meshblock.set_extent(m_sim_params.m_extent);
  m_meshblock.set_coord_system(m_sim_params.m_coord_system);
}

template <Dimension D>
void Simulation<D>::initialize() {
  // m_pGen.userInitFields(m_sim_params, m_meshblock);
  fieldBoundaryConditions(0.0);
  PLOGD << "Simulation initialized.";
}

void Simulation1D::verify() {
  m_sim_params.verify();
  m_meshblock.verify(m_sim_params);
  PLOGD << "Simulation prerun check passed.";
}
void Simulation2D::verify() {
  m_sim_params.verify();
  m_meshblock.verify(m_sim_params);
  PLOGD << "Simulation prerun check passed.";
}
void Simulation3D::verify() {
  m_sim_params.verify();
  m_meshblock.verify(m_sim_params);
  PLOGD << "Simulation prerun check passed.";
}

template <Dimension D>
void Simulation<D>::setIO(std::string_view infname, std::string_view outdirname) {
  m_sim_params.m_outputpath = outdirname;
  m_sim_params.m_inputfilename = infname;
}

void Simulation1D::printDetails() {
  m_sim_params.printDetails();
  m_meshblock.printDetails();
}
void Simulation2D::printDetails() {
  m_sim_params.printDetails();
  m_meshblock.printDetails();
}
void Simulation3D::printDetails() {
  m_sim_params.printDetails();
  m_meshblock.printDetails();
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
    faradayHalfsubstep(time);
    timers.stop(1);
  }

  {
    timers.start(2);
    fieldBoundaryConditions(time);
    timers.stop(2);
  }

  // particlePushSubstep(time);

  // depositSubstep(time);

  // BC particles
  // BC currents

  {
    timers.start(1);
    faradayHalfsubstep(time);
    timers.stop(1);
  }

  {
    timers.start(2);
    fieldBoundaryConditions(time);
    timers.stop(2);
  }

  {
    timers.start(1);
    ampereSubstep(time);
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
void Simulation<D>::mainloop() {
  PLOGD << "Simulation mainloop started.";

  unsigned long timax{static_cast<unsigned long>(m_sim_params.m_runtime / m_sim_params.m_timestep)};
  real_t time{0.0};
  for (unsigned long ti{0}; ti < timax; ++ti) {
    PLOGD << "t = " << time;
    step_forward(time);
    time += m_sim_params.m_timestep;
  }
  PLOGD << "Simulation mainloop finished.";
}

template <Dimension D>
void Simulation<D>::run(std::string_view infname, std::string_view outdirname) {
  this->setIO(infname, outdirname);
  this->initialize();
  this->verify();
  this->printDetails();
  this->mainloop();
  this->finalize();
}

} // namespace ntt

template class ntt::Simulation<ntt::ONE_D>;
template class ntt::Simulation<ntt::TWO_D>;
template class ntt::Simulation<ntt::THREE_D>;
