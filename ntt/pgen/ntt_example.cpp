#include "global.h"
#include "pgen.h"

#include "picsim.h"

#include <iostream>

class UserSimulation : public ntt::PICSimulation2D {
public:
  UserSimulation() : ntt::PICSimulation2D{ntt::POLAR_COORD} {}
  ~UserSimulation() = default;
  void initialize() override {
    ntt::PICSimulation2D::initialize();
    // user defined initialization goes here
    ex1.fillWith(2.0);
    ex2.fillWith(0.0);
    ex3.fillWith(0.0);
    bx1.fillWith(0.0);
    bx2.fillWith(-2.0);
    bx3.fillWith(0.0);
  }
  void finalize() override {}
};

ProblemGenerator::ProblemGenerator() { simulation = new UserSimulation(); }
