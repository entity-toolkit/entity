#include "global.h"
#include "constants.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include <cmath>

namespace ntt {

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
      mblock.bx3(i, j) = 1.0;
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
      mblock.particles[0].m_x1(p) = -1.4;
      mblock.particles[0].m_x2(p) = -1.4;
      mblock.particles[0].m_ux1(p) = 1.0;
      // positron
      mblock.particles[1].m_x1(p) = -1.4;
      mblock.particles[1].m_x2(p) = -1.4;
      mblock.particles[1].m_ux1(p) = 1.0;
  });
  mblock.particles[0].m_npart = 1;
  mblock.particles[1].m_npart = 1;

}

template <>
void ProblemGenerator<THREE_D>::userInitParticles(SimulationParams&,
                                                  Meshblock<THREE_D>&) {}

}
