#include "global.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include "problem_generator.hpp"
#include "init_fields.hpp"

#include <cmath>
#include <iostream>

namespace ntt {

  template <Dimension D, SimulationType S>
  ProblemGenerator<D, S>::ProblemGenerator(const SimulationParams&) {}

  // * * * * * * * * * * * * * * * * * * * * * * * *
  // Field initializers
  // . . . . . . . . . . . . . . . . . . . . . . . .
  template <>
  void ProblemGenerator<Dimension::ONE_D, SimulationType::GRPIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::ONE_D, SimulationType::GRPIC>&) {}

  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::GRPIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::GRPIC>& mblock) {

    Kokkos::parallel_for(
      "userInitFlds",
      NTTRange<Dimension::TWO_D>({mblock.i_min() - 1, mblock.j_min()}, {mblock.i_max(), mblock.j_max() + 1}),
      init_fields_potential<Dimension::TWO_D>(mblock, epsilon, A0, A1, A3));
  }

  template <>
  void ProblemGenerator<Dimension::THREE_D, SimulationType::GRPIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::THREE_D, SimulationType::GRPIC>&) {}
} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dimension::ONE_D, ntt::SimulationType::GRPIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::TWO_D, ntt::SimulationType::GRPIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::THREE_D, ntt::SimulationType::GRPIC>;

// * * * * * * * * * * * * * * * * * * * * * * * *
// Field boundary conditions
// . . . . . . . . . . . . . . . . . . . . . . . .

// template <>
// void ProblemGenerator<Dimension::ONE_D, SimulationType::GRPIC>::userBCFields(
//   const real_t&, const SimulationParams&, Meshblock<Dimension::ONE_D, SimulationType::GRPIC>&) {}

// template <>
// void ProblemGenerator<Dimension::TWO_D, SimulationType::GRPIC>::userBCFields(
// const real_t& time, const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::GRPIC>& mblock) {
//  using index_t = NTTArray<real_t**>::size_type;
//   (void) time;
//   Kokkos::parallel_for(
//     "userBcFlds_rmin",
//     NTTRange<Dimension::TWO_D>({mblock.i_min(), mblock.j_min()}, {mblock.i_min() + 1, mblock.j_max()}),
//     Lambda(index_t i, index_t j) {

//       // real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
//       // real_t j_ {static_cast<real_t>(j - N_GHOSTS)};
//       // real_t br_target  {userTargetField_br_cntrv(mblock, {i_, j_ + HALF})};

//       mblock.em0(i, j, em::ex3) = ZERO;
//       mblock.em0(i, j, em::ex2) = ZERO;
//       mblock.em0(i, j, em::bx1) = ZERO; //br_target;
//     });

//   Kokkos::parallel_for(
//     "userBcFlds_rmax",
//     NTTRange<Dimension::TWO_D>({mblock.i_max(), mblock.j_min()}, {mblock.i_max() + 1, mblock.j_max()}),
//     Lambda(index_t i, index_t j) {
//       mblock.em0(i, j, em::ex3) = ZERO;
//       mblock.em0(i, j, em::ex2) = ZERO;
//       mblock.em0(i, j, em::bx1) = ZERO;
//     });
// }

// template <>
// void ProblemGenerator<Dimension::THREE_D, SimulationType::GRPIC>::userBCFields(
//   const real_t&, const SimulationParams&, Meshblock<Dimension::THREE_D, SimulationType::GRPIC>&) {}
