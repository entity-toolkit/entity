#include "global.h"
#include "sim_params.h"
#include "meshblock.h"

#include "problem_generator.hpp"

namespace ntt  {
  template <>
  void ProblemGenerator<Dimension::ONE_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, const Meshblock<Dimension::ONE_D, ntt::SimulationType::PIC>&) {}

  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, const Meshblock<Dimension::TWO_D, ntt::SimulationType::PIC>&) {}

  template <>
  void ProblemGenerator<Dimension::THREE_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, const Meshblock<Dimension::THREE_D, ntt::SimulationType::PIC>&) {}
}

template struct ntt::ProblemGenerator<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;
