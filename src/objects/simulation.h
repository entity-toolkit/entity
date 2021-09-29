#ifndef OBJECTS_SIMULATION_H
#define OBJECTS_SIMULATION_H

#include "global.h"
#include "sim_params.h"
#include "meshblock.h"
#include "pgen.h"

#include <toml/toml.hpp>

#include <string>
#include <string_view>

namespace ntt {

template<template<typename T = std::nullptr_t> class D>
class Simulation {
  D<> m_dim;

  SimulationParams m_sim_params;
  Meshblock<D> m_meshblock;
  ProblemGenerator m_pGen;
public:
  Simulation(const toml::value &inputdata);
  ~Simulation() = default;
  void setIO(std::string_view infname, std::string_view outdirname);
  void initialize();
  void verify();
  void printDetails();
  void finalize();

  void mainloop();

  void faradayHalfsubstep(const real_t &time);
  void depositSubstep(const real_t &time);
  void ampereSubstep(const real_t &time);
  void addCurrentsSubstep(const real_t &time);
};

}

#endif
