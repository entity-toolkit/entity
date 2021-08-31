#include "global.h"
#include "sim.h"

#include <iostream>

class ProblemGenerator : public ntt::PICSimulation1D {
public:
  ProblemGenerator() {}
  ~ProblemGenerator() {}
  void initialize() {
    PICSimulation1D::initialize();
    m_title = "Some other silly name";
    std::cout << "TITLE: " << m_title << "\n";
  }
};

auto ntt_simulation = new ProblemGenerator();
