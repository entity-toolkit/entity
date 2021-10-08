#include "global.h"
#include "simulation.h"

#include <plog/Log.h>
#include <toml/toml.hpp>

namespace ntt {

Simulation1D::Simulation1D(const toml::value& inputdata)
    : Simulation<ONE_D>{inputdata}, m_meshblock{m_sim_params.m_resolution, m_sim_params.m_species} {
  // TODO: meshblock extent can be different from global one
  m_meshblock.set_extent(m_sim_params.m_extent);
  m_meshblock.set_coord_system(m_sim_params.m_coord_system);
}

Simulation2D::Simulation2D(const toml::value& inputdata)
    : Simulation<TWO_D>{inputdata}, m_meshblock{m_sim_params.m_resolution, m_sim_params.m_species} {
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

void Simulation1D::userInitialize() {
  m_pGen.userInitFields(m_sim_params, m_meshblock);
  fieldBoundaryConditions(0.0);
  m_pGen.userInitParticles(m_sim_params, m_meshblock);
  PLOGD << "Simulation initialized.";
}
void Simulation2D::userInitialize() {
  m_pGen.userInitFields(m_sim_params, m_meshblock);
  fieldBoundaryConditions(0.0);
  m_pGen.userInitParticles(m_sim_params, m_meshblock);
  PLOGD << "Simulation initialized.";
}
void Simulation3D::userInitialize() {
  m_pGen.userInitFields(m_sim_params, m_meshblock);
  fieldBoundaryConditions(0.0);
  m_pGen.userInitParticles(m_sim_params, m_meshblock);
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

}
