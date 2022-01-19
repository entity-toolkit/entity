#include "global.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include "ntt_boris_unit.hpp"

#include <cmath>

namespace ntt {

  template<Dimension D>
  ProblemGenerator<D>::ProblemGenerator(SimulationParams& sim_params) {
    UNUSED(sim_params);
    m_nx1 = readFromInput<int>(sim_params.m_inputdata, "problem", "nx1", 1);
    m_nx2 = readFromInput<int>(sim_params.m_inputdata, "problem", "nx2", 1);
    m_amplitude = readFromInput<real_t>(sim_params.m_inputdata, "problem", "amplitude", 1.0);
  }

  template <>
  void ProblemGenerator<ONE_D>::userInitFields(SimulationParams&,
                                               Meshblock<ONE_D>&) {}

  template <>
  void ProblemGenerator<TWO_D>::userInitFields(SimulationParams& sim_params,
                                               Meshblock<TWO_D>& mblock) {
    UNUSED(sim_params);
    using index_t = NTTArray<real_t**>::size_type;
    Kokkos::parallel_for(
      "userInitFlds", mblock.loopActiveCells(), Lambda(index_t i, index_t j) {
        mblock.em_fields(i, j, fld::ex2) = 0.1;
        mblock.em_fields(i, j, fld::bx3) = 1.0;
    });
  }

  template <>
  void ProblemGenerator<THREE_D>::userInitFields(SimulationParams&,
                                                 Meshblock<THREE_D>&) {}

  template <>
  void ProblemGenerator<ONE_D>::userInitParticles(SimulationParams&,
                                                  Meshblock<ONE_D>&) {}

  template <>
  void ProblemGenerator<TWO_D>::userInitParticles(SimulationParams& sim_params,
                                                  Meshblock<TWO_D>& mblock) {
    UNUSED(sim_params);
    using index_t = NTTArray<real_t*>::size_type;
    Kokkos::parallel_for(
      "userInitPrtls", NTT1DRange(0, 1), Lambda(index_t p) {
        // electron
        mblock.particles[0].x1(p) = 10;
        mblock.particles[0].x2(p) = 12;
        mblock.particles[0].dx1(p) = 0.3;
        mblock.particles[0].dx2(p) = 0.67;
        mblock.particles[0].ux1(p) = 1.0;
        mblock.particles[0].ux2(p) = 0.3;
        // positron
        mblock.particles[1].x1(p) = 7;
        mblock.particles[1].x2(p) = 30;
        mblock.particles[1].dx1(p) = 0.7;
        mblock.particles[1].dx2(p) = 0.35;
        mblock.particles[1].ux1(p) = -0.5;
        mblock.particles[1].ux2(p) = 0.5;
        // ion
        mblock.particles[2].x1(p) = 18;
        mblock.particles[2].x2(p) = 19;
        mblock.particles[2].x1(p) = 0.1;
        mblock.particles[2].x2(p) = 0.95;
        mblock.particles[2].ux1(p) = 2.0;
        mblock.particles[2].ux2(p) = 2.0;
    });
    mblock.particles[0].npart = 1;
    mblock.particles[1].npart = 1;
    mblock.particles[2].npart = 1;
  }

  template <>
  void ProblemGenerator<THREE_D>::userInitParticles(SimulationParams&,
                                                    Meshblock<THREE_D>&) {}

}

template struct ntt::ProblemGenerator<ntt::ONE_D>;
template struct ntt::ProblemGenerator<ntt::TWO_D>;
template struct ntt::ProblemGenerator<ntt::THREE_D>;
