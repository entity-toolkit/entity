#ifndef NTT_PGEN_H
#define NTT_PGEN_H

#include "global.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

class ProblemGenerator {

public:
  ProblemGenerator(SimulationParams &sim_params);
  ~ProblemGenerator() = default;

  template<template<typename T> class D>
  void userInitFields(SimulationParams &sim_params, Meshblock<D> &mblock);
};

}

#endif
