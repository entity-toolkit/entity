#include "global.h"
#include "pgen.h"

#include <iostream>

ProblemGenerator::ProblemGenerator() {
  simulation = new ntt::PICSimulation1D(ntt::CARTESIAN_COORD);
}
