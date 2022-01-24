#include "global.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include "problem_generator.hpp"

#include <cmath>

namespace ntt {

  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::PIC>& mblock) {
    using index_t = typename RealFieldND<Dimension::TWO_D, 6>::size_type;
    Kokkos::parallel_for(
      "userInitFlds", mblock.loopActiveCells(), Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::ex2) = 0.1;
        mblock.em(i, j, em::bx3) = 1.0;
      });
  }

  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::PIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::PIC>& mblock) {
    using index_t = const int;
    Kokkos::parallel_for(
      "userInitPrtls", NTTRange<Dimension::ONE_D>({0}, {1}), Lambda(index_t p) {
        // electron
        mblock.particles[0].i1(p) = 10;
        mblock.particles[0].i2(p) = 12;
        mblock.particles[0].dx1(p) = 0.3;
        mblock.particles[0].dx2(p) = 0.67;
        mblock.particles[0].ux1(p) = 1.0;
        mblock.particles[0].ux2(p) = 0.3;
        // positron
        mblock.particles[1].i1(p) = 7;
        mblock.particles[1].i2(p) = 30;
        mblock.particles[1].dx1(p) = 0.7;
        mblock.particles[1].dx2(p) = 0.35;
        mblock.particles[1].ux1(p) = -0.5;
        mblock.particles[1].ux2(p) = 0.5;
        // ion
        mblock.particles[2].i1(p) = 18;
        mblock.particles[2].i2(p) = 19;
        mblock.particles[2].dx1(p) = 0.1;
        mblock.particles[2].dx2(p) = 0.95;
        mblock.particles[2].ux1(p) = 2.0;
        mblock.particles[2].ux2(p) = 2.0;
      });
    mblock.particles[0].set_npart(1);
    mblock.particles[1].set_npart(1);
    mblock.particles[2].set_npart(1);
  }

  // 1D
  template <>
  void ProblemGenerator<Dimension::ONE_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::ONE_D, SimulationType::PIC>&) {}
  template <>
  void ProblemGenerator<Dimension::ONE_D, SimulationType::PIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dimension::ONE_D, SimulationType::PIC>&) {}

  // 3D
  template <>
  void ProblemGenerator<Dimension::THREE_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::THREE_D, SimulationType::PIC>&) {}
  template <>
  void ProblemGenerator<Dimension::THREE_D, SimulationType::PIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dimension::THREE_D, SimulationType::PIC>&) {}

} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;
