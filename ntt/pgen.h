#ifndef NTT_PGEN_H
#define NTT_PGEN_H

#include "sim.h"

class ProblemGenerator {
  ntt::AbstractSimulation *simulation;

public:
  ProblemGenerator();
  ~ProblemGenerator() = default;
  void start(int argc, char *argv[]) {
    simulation->parseInput(argc, argv);
    simulation->initialize();
    simulation->verify();
    simulation->printDetails();
    simulation->mainloop();
    simulation->finalize();
  }
};

#endif
