#include "global.h"
#include "pgen.h"

#include <iostream>

ProblemGenerator::ProblemGenerator() {
  simulation = new ntt::PICSimulation2D(ntt::POLAR_COORD);
}
