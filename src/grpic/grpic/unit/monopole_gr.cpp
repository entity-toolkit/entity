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
  void ProblemGenerator<Dim1, SimulationType::GRPIC>::userInitFields(
    const SimulationParams&, Meshblock<Dim1, SimulationType::GRPIC>&) {}

  template <>
  void ProblemGenerator<Dim2, SimulationType::GRPIC>::userInitFields(
    const SimulationParams&, Meshblock<Dim2, SimulationType::GRPIC>& mblock) {

    Kokkos::parallel_for("userInitFlds",
                         CreateRangePolicy<Dim2>({mblock.i1_min() - 1, mblock.i2_min()},
                                                 {mblock.i1_max(), mblock.i2_max() + 1}),
                         initFieldsFromVectorPotential<Dim2>(*this, mblock, epsilon));
  }

  template <>
  void ProblemGenerator<Dim3, SimulationType::GRPIC>::userInitFields(
    const SimulationParams&, Meshblock<Dim3, SimulationType::GRPIC>&) {}
} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dim1, ntt::SimulationType::GRPIC>;
template struct ntt::ProblemGenerator<ntt::Dim2, ntt::SimulationType::GRPIC>;
template struct ntt::ProblemGenerator<ntt::Dim3, ntt::SimulationType::GRPIC>;

// !LEGACY
// * * * * * * * * * * * * * * * * * * * * * * * *
// Field boundary conditions
// . . . . . . . . . . . . . . . . . . . . . . . .
// template <>
// void ProblemGenerator<Dim1, SimulationType::GRPIC>::userBCFields(
//   const real_t&, const SimulationParams&, Meshblock<Dim1, SimulationType::GRPIC>&) {}

// template <>
// void ProblemGenerator<Dim2, SimulationType::GRPIC>::userBCFields(
// const real_t& time, const SimulationParams&, Meshblock<Dim2, SimulationType::GRPIC>& mblock)
// {
//

//  Kokkos::parallel_for(
//     "userBcFlds_rmin",
//     // CreateRangePolicy<Dim2>({0, mblock.i2_min()}, {mblock.i1_min() + 1,
//     mblock.i2_max()}), CreateRangePolicy<Dim2>({mblock.i1_min(), mblock.i2_min()},
//     {mblock.i1_min() + 1, mblock.i2_max()}), Lambda(index_t i, index_t j) {

//       real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
//       real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};
//       real_t br_target  {userTargetField_br_cntrv(mblock, {i_, j_ + HALF})};

//       mblock.em0(i, j, em::ex3) = ZERO;
//       mblock.em0(i, j, em::ex2) = ZERO;
//       mblock.em0(i, j, em::bx1) = br_target;
//     });

//   Kokkos::parallel_for(
//     "userBcFlds_rmax",
//     CreateRangePolicy<Dim2>({mblock.i1_max(), mblock.i2_min()}, {mblock.i1_max() + 1,
//     mblock.i2_max()}), Lambda(index_t i, index_t j) {
//       mblock.em0(i, j, em::ex3) = ZERO;
//       mblock.em0(i, j, em::ex2) = ZERO;
//       mblock.em0(i, j, em::bx1) = ZERO;
//     });
// }

// template <>
// void ProblemGenerator<Dim3, SimulationType::GRPIC>::userBCFields(
//   const real_t&, const SimulationParams&, Meshblock<Dim3, SimulationType::GRPIC>&) {}
