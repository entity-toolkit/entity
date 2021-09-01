#include "global.h"
#include "pgen.h"

#include <iostream>

ProblemGenerator::ProblemGenerator() : Simulation(ntt::ONE_D, ntt::CARTESIAN_COORD, ntt::PIC_SIM) {
  ex1 = new ntt::arrays::OneDArray<ntt::real_t>(100);
  ex1->fillWith(1.0);
  std::cout << "size in bytes: " << ex1->getSizeInBytes() << "\n";
}
