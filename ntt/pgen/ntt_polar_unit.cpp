#include "global.h"
#include "constants.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include <cmath>

namespace ntt {

template<Dimension D>
ProblemGenerator<D>::ProblemGenerator(SimulationParams& sim_params) {
  UNUSED(sim_params);
}

// * * * * * * * * * * * * * * * * * * * * * * * *
// Field initializers
// . . . . . . . . . . . . . . . . . . . . . . . .
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
      mblock.em_fields(i, j, fld::ex1) = 0.0;
      mblock.em_fields(i, j, fld::ex2) = 0.0;
      mblock.em_fields(i, j, fld::ex3) = 0.0;
      mblock.em_fields(i, j, fld::bx1) = 0.0;
      mblock.em_fields(i, j, fld::bx2) = 0.0;
      mblock.em_fields(i, j, fld::bx3) = 0.0;
  });
}

template <>
void ProblemGenerator<THREE_D>::userInitFields(SimulationParams&,
                                               Meshblock<THREE_D>&) {}

// * * * * * * * * * * * * * * * * * * * * * * * *
// Field boundary conditions
// . . . . . . . . . . . . . . . . . . . . . . . .
template <>
void ProblemGenerator<ONE_D>::userBCFields_x1min(SimulationParams&,
                                                 Meshblock<ONE_D>&) {}

template <>
void ProblemGenerator<TWO_D>::userBCFields_x1min(SimulationParams& sim_params,
                                                 Meshblock<TWO_D>& mblock) {
  UNUSED(sim_params);
  using index_t = NTTArray<real_t**>::size_type;
  Kokkos::parallel_for(
    "userBcFlds", mblock.loopX1MinCells(), Lambda(index_t i, index_t j) {
      mblock.em_fields(i, j, fld::bx3) = 1.0;
  });
}

template <>
void ProblemGenerator<THREE_D>::userBCFields_x1min(SimulationParams&,
                                                   Meshblock<THREE_D>&) {}

template <>
void ProblemGenerator<ONE_D>::userBCFields_x1max(SimulationParams&,
                                                 Meshblock<ONE_D>&) {}

template <>
void ProblemGenerator<TWO_D>::userBCFields_x1max(SimulationParams& sim_params,
                                                 Meshblock<TWO_D>& mblock) {
  UNUSED(sim_params);
  using index_t = NTTArray<real_t**>::size_type;
  Kokkos::parallel_for(
    "userBcFlds", mblock.loopX1MaxCells(), Lambda(index_t i, index_t j) {
      mblock.em_fields(i, j, fld::bx3) = 1.0;
  });
}

template <>
void ProblemGenerator<THREE_D>::userBCFields_x1max(SimulationParams&,
                                                  Meshblock<THREE_D>&) {}


}
template struct ntt::ProblemGenerator<ntt::ONE_D>;
template struct ntt::ProblemGenerator<ntt::TWO_D>;
template struct ntt::ProblemGenerator<ntt::THREE_D>;
