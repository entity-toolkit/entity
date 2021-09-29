#ifndef OBJECTS_SIM_PARAMS_H
#define OBJECTS_SIM_PARAMS_H

#include "global.h"

#include <toml/toml.hpp>

#include <vector>
#include <string_view>

namespace ntt {

class SimulationParams {
  std::string_view m_inputfilename;
  std::string_view m_outputpath;
  toml::value m_inputdata;

  SimulationType m_simtype{UNDEFINED_SIM};

  std::string m_title;
  real_t m_timestep;
  real_t m_runtime;

  // independent params
  real_t m_ppc0;
  real_t m_larmor0;
  real_t m_skindepth0;

  // dependent params
  real_t m_sigma0;
  real_t m_charge0;

  CoordinateSystem m_coord_system{UNDEFINED_COORD};
  std::vector<real_t> m_extent;
  std::vector<std::size_t> m_resolution;
  std::vector<BoundaryCondition> m_boundaries;
public:
  SimulationParams(const toml::value &inputdata, short dim);
  ~SimulationParams() = default;

  template<template<typename T> class D>
  friend class Simulation;

  friend class ProblemGenerator;
};

}

#endif
