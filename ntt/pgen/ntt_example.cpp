#include "global.h"
#include "pgen.h"

#include <iostream>

class UserSimulation : public ntt::PICSimulation2D {
public:
  UserSimulation() : ntt::PICSimulation2D{ntt::POLAR_COORD, ntt::BORIS_PUSHER} {}
  ~UserSimulation() = default;
  void initialize() override {
    ntt::PICSimulation2D::initialize();
    m_domain.set_boundaries({ntt::PERIODIC_BC, ntt::PERIODIC_BC});
    // user defined initialization goes here
    std::cout << m_species.getSizeInBytes() << " B\n";
  }
  void finalize() override {}
};

ProblemGenerator::ProblemGenerator() { simulation = new UserSimulation(); }
