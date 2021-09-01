#include "global.h"
#include "pgen.h"

#include <iostream>

class UserSimulation : public ntt::PICSimulation2D {
public:
  UserSimulation() : ntt::PICSimulation2D{ntt::POLAR_COORD, ntt::BORIS_PUSHER} {}
  ~UserSimulation() = default;
  void initialize() {
    ntt::PICSimulation2D::initialize();
    // user defined initialization goes here
  }
  void finalize() {}
};

ProblemGenerator::ProblemGenerator() {
  simulation = new UserSimulation();
}
