#include "global.h"
#include "simulation.h"
#include "sim_params.h"
#include "meshblock.h"
#include "pgen.h"

#include <plog/Log.h>

#include <cassert>

namespace ntt {

template<template<typename T> class D>
Simulation<D>::Simulation(int argc, char *argv[]) : m_dim{}, m_sim_params{argc, argv, m_dim.dim}, m_meshblock{m_sim_params.m_resolution}, m_pGen{m_sim_params} {}

template<template<typename T> class D>
void Simulation<D>::initialize() {
  m_pGen.userInitFields(m_sim_params, m_meshblock);
  PLOGD << "Simulation initialized.";
}

template<template<typename T> class D>
void Simulation<D>::verify() {
  assert(m_sim_params.m_simtype != UNDEFINED_SIM);
  assert(m_sim_params.m_coord_system != UNDEFINED_COORD);
  for (auto & b : m_sim_params.m_boundaries) {
    assert(b != UNDEFINED_BC);
  }
  // TODO: maybe some other tests
  PLOGD << "Simulation prerun check passed.";
}

template<template<typename T> class D>
void Simulation<D>::printDetails() {
  PLOGI << "[Simulation details]";
  PLOGI << "   title: " << m_sim_params.m_title;
  PLOGI << "   type: " << stringifySimulationType(m_sim_params.m_simtype);
  PLOGI << "   total runtime: " << m_sim_params.m_runtime;
  PLOGI << "   dt: " << m_sim_params.m_timestep << " [" << static_cast<int>(m_sim_params.m_runtime / m_sim_params.m_timestep) << " steps]";

  PLOGI << "[domain]";
  // PLOGI << "   dimension: " << stringifyDimension(m_sim_params.m_dimension);
  PLOGI << "   coordinate system: " << stringifyCoordinateSystem(m_sim_params.m_coord_system);

  std::string bc {"   boundary conditions: { "};
  for (auto & b : m_sim_params.m_boundaries) {
    bc += stringifyBoundaryCondition(b) + " x ";
  }
  bc.erase(bc.size() - 3);
  bc += " }";
  PLOGI << bc;

  std::string res {"   resolution: { "};
  for (auto & r : m_sim_params.m_resolution) {
    res += std::to_string(r) + " x ";
  }
  res.erase(res.size() - 3);
  res += " }";
  PLOGI << res;

  std::string ext {"   extent: "};
  for (std::size_t i{0}; i < m_sim_params.m_extent.size(); i += 2) {
    ext += "{" + std::to_string(m_sim_params.m_extent[i]) + ", " + std::to_string(m_sim_params.m_extent[i + 1]) + "} ";
  }
  PLOGI << ext;
}

template<template<typename T> class D>
void Simulation<D>::finalize() {
  PLOGD << "Simulation finalized.";
}

template class ntt::Simulation<ntt::One_D>;
template class ntt::Simulation<ntt::Two_D>;
template class ntt::Simulation<ntt::Three_D>;

}
