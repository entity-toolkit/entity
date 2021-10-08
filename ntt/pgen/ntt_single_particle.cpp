#include "global.h"
#include "constants.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include <cmath>

namespace ntt {

template <>
void ProblemGenerator<ONE_D>::userInitFields(SimulationParams&,
                                             Meshblock1D&) {}

template <>
void ProblemGenerator<TWO_D>::userInitFields(SimulationParams& sim_params,
                                             Meshblock2D& mblock) {
  UNUSED(sim_params);
  using size_type = NTTArray<real_t**>::size_type;
  Kokkos::parallel_for(
    "userInitFlds", mblock.loopActiveCells(), Lambda(size_type i, size_type j) {
      mblock.bx3(i, j) = 1.0;
  });
}

template <>
void ProblemGenerator<THREE_D>::userInitFields(SimulationParams&,
                                               Meshblock3D&) {}

template <>
void ProblemGenerator<ONE_D>::userInitParticles(SimulationParams&,
                                                Meshblock1D&) {}

template <>
void ProblemGenerator<TWO_D>::userInitParticles(SimulationParams& sim_params,
                                                Meshblock2D& mblock) {
  UNUSED(sim_params);
  using size_type = NTTArray<real_t*>::size_type;
  Kokkos::parallel_for(
    "userInitPrtls", NTT1DRange(0, 1), Lambda(size_type p) {
      // electron
      mblock.particles[0].m_x1(p) = 1.4;
      mblock.particles[0].m_x2(p) = -1.4;
      // positron
      mblock.particles[1].m_x1(p) = 1.4;
      mblock.particles[1].m_x2(p) = -1.4;
  });
  mblock.particles[0].m_npart = 1;
  mblock.particles[1].m_npart = 1;

}

template <>
void ProblemGenerator<THREE_D>::userInitParticles(SimulationParams&,
                                                  Meshblock3D&) {}

}
